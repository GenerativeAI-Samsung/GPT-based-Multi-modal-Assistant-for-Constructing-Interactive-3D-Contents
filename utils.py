import torch
import torch.nn as nn
import torch.nn.functional as F

import g4f
import asyncio
import random


def split_answer_from_respone(respone):
    list_answer = []
    for res in respone:
        answer1 = res.split('Respone:')[1]
        if ('object_list = [' in answer1):
            answer2 = answer1.split('object_list = [')[1]
            answer3 = answer2.split(']')[0]
            final_answer = 'object_list = [' + answer3 + ']'
            list_answer.append(final_answer)
        else:
            list_answer.append("object_list = []")
    return list_answer

def interact_with_lm(tokenizer, model, prompt, setting):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    print(f"len(prompt): {len(prompt)}")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    if (setting == "peft_model"):
        outputs = model.generate(**inputs, max_length=4096, output_hidden_states=True, return_dict_in_generate=True)
    if (setting == "base_model"):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=4096, output_hidden_states=True, return_dict_in_generate=True)

    # Extract last hidden state
    # Last hidden state is a tuple of sequence_length. 
    #   Each items is a tensor of [batch_size, feature, hidden_dim]
    if hasattr(outputs, 'decoder_hidden_states'):
        last_hidden_state = outputs.decoder_hidden_states[-1]
    else:
        last_hidden_state = outputs.hidden_states[-1]

    # Decode the generated tokens back to text
    decoded_outputs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
    
    return decoded_outputs, last_hidden_state

async def process_api_request(request, index):
    while True:
        try:
            await asyncio.sleep(random.randint(10, 20))
            print(request)
            print(f"Started API request of index: {index}.")
            response = await g4f.ChatCompletion.create_async(
                model="gpt-4o",
                messages=[{"role": "user", "content": request}],
            )
            if len(response) == 0:
                continue
            try: 
              exec(response)
              print(f"Completed API request of index: {index}")
              return response
            except:
                warining = """
    -------------------------------------------------------------------------
    YOUR PREVIOUS ANSWER DID NOT REPSONE IN RIGHT FORMAT!
    REMEMBER TO STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    rewarding_score = [{{"name": criteria1, "score": score1, "description": description1}}, 
                        {{"name": criteria2, "score": score2, "description": description2}},
                        ...]
    ------------------------------------------------------------------------
    """
                request = request + warining
                continue

        except Exception as e:
            print(f"Request of index {index} - Error: {str(e)}")
            await asyncio.sleep(10)

async def generate_reward_score_from_api(prompt):
    #async with Client() as session:
        tasks = []
        for index, request in enumerate(prompt):
            tasks.append(process_api_request(request, index))
        return await asyncio.gather(*tasks, return_exceptions=True)

def RLAIF_loss_fuction(score_response, last_hidden_state, base_last_hidden_state, batch_size, beta=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_score = []
    # Caculate the average rewarding score 
    
    # Local variables
    local_vars = {}
    for item in score_response:
        exec(item, {}, local_vars)
        for reward_item in local_vars['rewarding_score']:
            reward_score += reward_item['score']
            print(f"item: {reward_item},reward_score: {reward_score}")
        reward_score.append(torch.tensor(reward_score / (10 * len(local_vars['rewarding_score']))))

    reward_score = torch.stack(reward_score, dim=0).to(device)
    
    print(f"final score: {reward_score}")

    # KL diverage 
    kl_loss = nn.KLDivLoss(reduction="none", log_target=True)

    # Stack both last_hidden_state
    last_hidden_state = torch.stack(list(last_hidden_state)).to(device)
    base_last_hidden_state = torch.stack(list(base_last_hidden_state)).to(device)

    # then transpose
    last_hidden_state = torch.transpose(last_hidden_state, 0, 1)
    base_last_hidden_state = torch.transpose(base_last_hidden_state, 0, 1)

    # Add padding if two hidden state is not in same length
    padding = torch.zeros(4, 1, 1, 4096)
    if (last_hidden_state.shape[1] < base_last_hidden_state.shape[1]):
        last_hidden_state = torch.cat((last_hidden_state, padding), 1)
    elif (last_hidden_state.shape[1] > base_last_hidden_state.shape[1]):
        base_last_hidden_state = torch.cat((base_last_hidden_state, padding), 1)
    
    # Caculate distribution on last_hidden_state and base_last_hidden_state
    agent_log_distribution = F.log_softmax(last_hidden_state, dim=-1)
    base_log_distribution = F.log_softmax(item, dim=-1)
    
    kl_loss_value = kl_loss(agent_log_distribution.to(device), base_log_distribution.to(device))

    kl_loss_average = torch.squeeze(kl_loss_value, dim=-2).mean(dim=-1).mean(dim=-1)

    print(f"kl_loss_average: {kl_loss_average}")

    beta = torch.tensor(beta).to(device)

    total_loss = (1 - beta) * reward_score - beta * kl_loss_average
    total_loss = total_loss.mean() 
    print(f"total_loss: {total_loss}")

    return total_loss
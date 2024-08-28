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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids

    batch_size, seq_length = inputs.shape
    num_tokens_to_generate = 2048
    # Generate the output
    if (setting == "peft_model"):
        for _ in range(num_tokens_to_generate):
            # Perform forward pass
            outputs = model(inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits
            print(logits.shape)
            # Get the last token logits and predict the next token
            next_token_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the predicted tokens to the input_ids
            inputs = torch.cat([inputs, next_token_ids], dim=-1)
    if (setting == "base_model"):
        with torch.no_grad():
            for _ in range(num_tokens_to_generate):
                # Perform forward pass
                outputs = model(inputs, output_hidden_states=False, return_dict=True)
                logits = outputs.logits
                print(logits.shape)

                # Get the last token logits and predict the next token
                next_token_logits = logits[:, -1, :]
                next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # Append the predicted tokens to the input_ids
                inputs = torch.cat([inputs, next_token_ids], dim=-1)
    
    decoded_outputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)    
    return decoded_outputs, outputs.logits[ :, seq_length: , : ], outputs

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
        temp = 0
        for reward_item in local_vars['rewarding_score']:
            temp += reward_item['score']
            print(f"item: {reward_item},reward_score: {temp}")
        reward_score.append(torch.tensor(temp / (10 * len(local_vars['rewarding_score']))))

    reward_score = torch.stack(reward_score, dim=0).to(device)
    
    print(f"final score: {reward_score}")

    # KL diverage 
    kl_loss = nn.KLDivLoss(reduction="none", log_target=True)

    # Add padding if two hidden state is not in same length
    padding = torch.zeros(batch_size, 1, 128258)
    if (last_hidden_state.shape[1] < base_last_hidden_state.shape[1]):
        last_hidden_state = (torch.cat((last_hidden_state, padding), 1) for i in range(base_last_hidden_state.shape[1] - last_hidden_state.shape[1]))
    elif (last_hidden_state.shape[1] > base_last_hidden_state.shape[1]):
        base_last_hidden_state = (torch.cat((base_last_hidden_state, padding), 1) for i in range(last_hidden_state.shape[1] - base_last_hidden_state.shape[1]))
    
    print(f"last_hidden_state require_grad: {last_hidden_state.requires_grad} {last_hidden_state.shape}")
    print(f"base_last_hidden_state require_grad: {base_last_hidden_state.requires_grad} {base_last_hidden_state.shape}")

    # Caculate distribution on last_hidden_state and base_last_hidden_state
    agent_log_distribution = F.log_softmax(last_hidden_state, dim=-1)
    base_log_distribution = F.log_softmax(base_last_hidden_state, dim=-1)
    
    kl_loss_value = kl_loss(agent_log_distribution.to(device), base_log_distribution.to(device))

    kl_loss_average = torch.squeeze(kl_loss_value, dim=-2).mean(dim=-1).mean(dim=-1)

    print(f"kl_loss_average: {kl_loss_average}")

    beta = torch.tensor(beta, requires_grad=True).to(device)

    total_loss = beta * kl_loss_average -(1 - beta) * reward_score  
    total_loss = total_loss.mean() 
    print(f"total_loss: {total_loss}")

    return total_loss
import torch
import torch.nn as nn
from g4f.client import Client

def split_answer_from_respone(respone):
    answer = respone.split('Respone:')
    return answer[1]

def interact_with_lm(tokenizer, model, prompt, setting):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    if (setting == "peft_model"):
        outputs = model.generate(**inputs, max_length=4096, output_hidden_states=True, return_dict_in_generate=True)
    if (setting == "base_model"):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=4096, output_hidden_states=True, return_dict_in_generate=True)

    # Extract all hidden states
    if hasattr(outputs, 'decoder_hidden_states'):
        last_hidden_states = outputs.decoder_hidden_states[-1]
    else:
        last_hidden_states= outputs.hidden_states[-1]

    # Decode the generated tokens back to text
    outputs = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return outputs, last_hidden_states

def generate_reward_score_from_api(prompt):
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def RLAIF_loss_fuction(rewarding_score, last_hidden_state, base_last_hidden_state, beta=0.2):
    # Caculate the average rewarding score 
    for item in rewarding_score:
        reward_score = item['score']
        print(f"item: {item},reward_score: {reward_score}")
    reward_score = torch.tensor(reward_score / (10 * len(rewarding_score))) 
    
    print(f"final score: {reward_score}")

    # KL diverage loss
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss_sum = torch.tensor(0)

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    reward_score.to(device)
    kl_loss_sum.to(device)

    if (len(last_hidden_state) >= len(base_last_hidden_state)):
        number_of_pairs = torch.tensor(len(last_hidden_state)).to(device)
        for i, item in enumerate(last_hidden_state):
            if (i < len(base_last_hidden_state)):
                kl_loss_sum += kl_loss(item, base_last_hidden_state[i])
            else:
                padding_token = torch.zeros(1, 1, 4096)
                kl_loss_sum += kl_loss(item, padding_token)
    else:
        number_of_pairs = torch.tensor(len(base_last_hidden_state)).to(device)
        for i, item in enumerate(base_last_hidden_state):
            if (i < len(last_hidden_state)):
                kl_loss_sum += kl_loss(last_hidden_state[i], item)
            else:
                padding_token = torch.zeros(1, 1, 4096)
                kl_loss_sum += kl_loss(padding_token, item)
    
    kl_loss_average = kl_loss_sum / number_of_pairs

    print(f"kl_loss_average: {kl_loss_average}")

    total_loss = (1 - beta) * reward_score + beta * kl_loss_average
    return total_loss
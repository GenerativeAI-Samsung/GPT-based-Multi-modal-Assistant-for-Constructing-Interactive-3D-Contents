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
        all_hidden_states = outputs.decoder_hidden_states
    else:
        all_hidden_states = outputs.hidden_states

    # Decode the generated tokens back to text
    outputs = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return outputs, all_hidden_states

def generate_reward_score_from_api(prompt):
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def RLAIF_loss_fuction(rewarding_score, all_hidden_state, base_all_hidden_state, beta=0.2):
    # Caculate the average rewarding score 
    for item in rewarding_score:
        reward_score = item['score']
        print(f"item: {item},reward_score: {reward_score}")
    reward_score = torch.tensor(reward_score / (10 * len(rewarding_score))) 

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_score.to(device)
    all_hidden_state[0].to(device)
    base_all_hidden_state[0].to(device)

    print(f"last_hidden_state[0].shape: {all_hidden_state[0].shape}")
    print(f"base_last_hidden_state[0].shape: {base_all_hidden_state[0].shape}")

    # Caculate KL loss between last_hidden_state and base_last_hidden_state
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss_output = (all_hidden_state[0], base_all_hidden_state[0])
    
    print(f"kl_loss_output[0].shape: {kl_loss_output[0].shape}")

    total_loss = (1 - beta) * reward_score + beta * kl_loss_output
    return total_loss
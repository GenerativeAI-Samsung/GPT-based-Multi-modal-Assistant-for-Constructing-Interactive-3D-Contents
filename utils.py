import torch
from g4f.client import Client

def split_answer_from_respone(respone):
    answer = respone.split('Respone:')
    return answer

def interact_with_lm(tokenizer, model, prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    outputs = model.generate(**inputs, max_length=4096)

    # Decode the generated tokens back to text
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return outputs

def generate_reward_score_from_api(prompt):
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
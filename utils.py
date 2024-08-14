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

bl_info = {
    "name": "GPT Blender",
    "author": "Embedded Networking Laboratory",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "",
    "description": "Instant 3D content generation via LLM",
    "warning": "",
    "doc_url": "",
    "category": "3D content generation",
}

def generate_response(self, history, new_prompt):
    messages = [
        {"role": "system", "content": "You're an assistant to help in Blender code. Your job is help user writting a full Blender program in Python"},
    ]
    messages.extend([{"role": entry.role, "content": entry.content} for entry in history])
    messages.append(new_prompt)
    
    response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    
    response_content = response.choices[0].message.content

    user_entry = history.add()
    user_entry.role = "user"
    user_entry.content = new_prompt['content']

    assistant_entry = history.add()
    assistant_entry.role = "assistant"
    assistant_entry.content = response_content
    
    return response_content

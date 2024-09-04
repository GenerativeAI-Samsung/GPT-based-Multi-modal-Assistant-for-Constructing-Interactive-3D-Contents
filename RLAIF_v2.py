import warnings
import asyncio
import g4f
import random

from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return {"input_ids": self.data[idx]["respone"]}
  
def split_answer_from_respone(respone):
    list_answer = []
    try: 
      for res in respone:
          answer1 = res.split('Respone:')[1]
          if ('object_list = [' in answer1):
              answer2 = answer1.split('object_list = [')[1]
              answer3 = answer2.split(']')[0]
              final_answer = 'object_list = [' + answer3 + ']'
              list_answer.append(final_answer)
          else:
              list_answer.append("object_list = []")
    except:
      warnings.warn(f"Code execution failed: {respone}")
    return list_answer

def prompt_reward(criteria, answer_format, prompt, response):
    rewarding_prompts = []
    for prom, res in zip(prompt, response):
        formatted_criteria = "".join(f"\t-{item['name']}: {item['description']}\n" for item in criteria)
        rewarding_prompt = f"""
    You are an evaluator. Your task is to grade the response provided by the responder to the user's request based on specific criteria, using a 100-point scale.
    The criteria include:
    {formatted_criteria}

    The responder's answer is formatted as:
    {answer_format}

    User's request: "{prom}"

    Responder's answer: "{res}"

    After determining your answer, structure them in this format:
    rewarding_score = [{{"name": criteria1, "score": score1, "description": description1}},
                        {{"name": criteria2, "score": score2, "description": description2}},
                        ...]

    Avoid using normal text; format your response strictly as specified above.
    -------------------------------------------------------------------------
    REMEMBER TO STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    rewarding_score = [{{"name": criteria1, "score": score1, "description": description1}},
                        {{"name": criteria2, "score": score2, "description": description2}},
                        ...]
    ------------------------------------------------------------------------
    """
        rewarding_prompts.append(rewarding_prompt)
    return rewarding_prompts

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

def step1(user_request):
    step1_prompts = []

    step1_answer_format = """
object_list = [
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
"""
    for request in user_request:
        step1_prompt = f"""
    You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description.
    Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets).

    Natural language description: "{request}"

    After listing the assets, structure them in this format:
    {step1_answer_format}

    Avoid using normal text; format your response strictly as specified above.
    """
        step1_prompt += f"""
    -------------------------------------------------------------------------
    REMEMBER TO STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
"""
        step1_prompt += "\nRespone:"
        step1_prompts.append(step1_prompt)
    return step1_prompts

def custom_collate_fn(batch):
    return batch

class CustomTrainer(Trainer):
  def __init__(self, custom_base_model, custome_tokenizer, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_base_model = custom_base_model
    self.custome_tokenizer = custome_tokenizer
    self.criteria = [
        {'name': 'Accuracy',
            'description': 'Are the objects identified fully and accurately?'},
        {'name': 'Coverage',
            'description': 'Is any necessary object for the scene missing?'},
        {'name': 'Relevance to Requirements',
            'description': 'Do the objects align with the goals and requirements of the scene?'}
    ]
    self.answer_format = """
object_list = [
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
"""
  def compute_loss(self, model, inputs, return_outputs=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate the content
    modified_inputs = step1(inputs)

    tokenized_inputs = self.custome_tokenizer(modified_inputs, return_tensors="pt", padding=True).to(device)
    
    # Generate content
    outputs = model.generate(**tokenized_inputs, max_length=1024, output_hidden_states=True, return_dict_in_generate=True)
    content_texts = self.custome_tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    list_answer = split_answer_from_respone(content_texts)
    del outputs

    # Generate last hidden state of base model
    outputs = self.custom_base_model.generate(**tokenized_inputs, max_length=1268, output_hidden_states=True, return_dict_in_generate=True)
    base_state = outputs.hidden_states[-1].clone().detach()
    del outputs

    # Creating Rewarding Prompt
    rewarding_prompt = prompt_reward(model.criteria, model.answer_format, modified_inputs, list_answer)

    # Get reward score through API
    score_response = asyncio.run(generate_reward_score_from_api(prompt=rewarding_prompt))

    # Caculate reward score
    reward_score = []
    local_vars = {}
    for item in score_response:
      exec(item, {}, local_vars)
      temp = 0
      for reward_item in local_vars['rewarding_score']:
        temp += reward_item['score']
        print(f"item: {reward_item},reward_score: {temp}")
      reward_score.append(torch.tensor(temp / (10 * len(local_vars['rewarding_score']))))

    reward_score = torch.stack(reward_score, dim=0).to(device)

    # Caculate forward of peft model
    content_inputs = self.tokenizer(content_texts, return_tensors="pt", padding=True)
    state = model(content_inputs)[-1]

    # Add padding if two hidden state is not in same length
    padding = torch.zeros(4, 1, 1, 4096).to(device)
    if (state.shape[1] < base_state.shape[1]):
        state = torch.cat((state, padding), 1)
    elif (state.shape[1] > base_state.shape[1]):
        base_state = torch.cat((base_state, padding), 1)

    # Define loss kl loss
    kl_loss = nn.KLDivLoss(reduction="none", log_target=True)

    # Caculate kl loss value
    agent_log_distribution = F.log_softmax(state, dim=-1)
    base_log_distribution = F.log_softmax(base_state, dim=-1)

    kl_loss_value = kl_loss(agent_log_distribution.to(device), base_log_distribution.to(device))

    # Combine loss
    beta = torch.tensor(0.2, requires_grad=True).to(device)
    total_loss = torch.log((beta * kl_loss_average - (1 - beta) * reward_score) * state)
    return total_loss

  def get_train_dataloader(self):
    # Create DataLoader with the custom collate function
    return DataLoader(
      self.train_dataset,
      batch_size=self.args.per_device_train_batch_size,
      collate_fn=custom_collate_fn,
      shuffle=True,
      num_workers=self.args.dataloader_num_workers
      )

MODEL_ID = "LoftQ/Phi-3-mini-4k-instruct-4bit-64rank"

# # Load tokenizer and model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
# model = CustomModel(base_model=base_model, MODEL_ID=MODEL_ID, option="1")
# model.custom_tokenizer()
peft_model = PeftModel.from_pretrained(
  base_model,
  MODEL_ID,
  subfolder="loftq_init",
  is_trainable=True)
tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        model_max_length=1268,
        padding_side="right",
        use_fast=False)
# Load Dataset
# Read data from json file
f = open("/content/train_examples.json")
train_data = json.load(f)
ds = CustomDataset(train_data)

default_args = {
    'output_dir': './results',
    'num_train_epochs': 3,
    'logging_dir': './logs',
    # Add other default arguments here
}

# Training Argument
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    **default_args
)

trainer = CustomTrainer(model=peft_model,
                        args=training_args,
                        train_dataset=ds,
                        custom_base_model=base_model,
                        custome_tokenizer=tokenizer)
result = trainer.train()




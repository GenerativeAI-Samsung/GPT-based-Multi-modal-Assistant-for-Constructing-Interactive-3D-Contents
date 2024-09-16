import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from typing import Dict

import g4f
import asyncio
import random
import gc
import json

def smart_tokenizer_and_embedding_resize_base_model(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    base_model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    base_model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = base_model.get_input_embeddings().weight.data
        output_embeddings = base_model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    base_model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    base_model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def step1_preprocess_data(batch):
    processed_batch = []

    step1_answer_format = """
object_list = [
  {"name": x1, "description": y1},
  {"name": x2, "description": y2},
  {"name": x3, "description": y3},
  ...
]
Each asset is described with a concise name (x) and a detailed visual description (y).
"""
    for sample in batch:
        processed_sample = f"""
    You are an assistant for developing multiple Blender scripts to create scenes for diverse animation projects from natural description. 
    Your job is to list the assets individually, ensuring each is a single unit (avoiding composite sets). 

    Natural language description: "{sample['respone']}"    
    
    After listing the assets, structure them in this format:
    {step1_answer_format}

    Avoid using normal text; format your response strictly as specified above.
    """
        processed_sample += f"""
    -------------------------------------------------------------------------
    REMEMBER TO ADVOID USING NORMAL AND STRUCTURE YOUR RESPONE STRICTLY AS SPECIFIC AS:
    {step1_answer_format}
    ------------------------------------------------------------------------
"""
        processed_sample += "\nRespone:"
        processed_batch.append(processed_sample)
    
    return {"processed_batch": processed_batch, 
            "answer_format": step1_answer_format}

def step1_crop_respone(batch):
    cropped_respone_batch = []
    for respone in batch:
        temp1 = respone.split('\nRespone:', 1)[1]
        if ('object_list = [' in temp1):
            temp2 = temp1.split('object_list = [')[1]
            temp3 = temp2.split(']')[0]
            temp = 'object_list = [' + temp3 + ']'
            print(f"respone: {temp}")
            cropped_respone_batch.append(temp)
        else:
            cropped_respone_batch.append("object_list = []")
            print(f"respone: {temp}")
    return cropped_respone_batch

def craft_rewarding_prompt(processed_batch, cropped_respone_batch, scoring_criterias):
    rewarding_prompts = []
    formatted_criteria = "".join(f"\t-{item['name']}: {item['description']}\n" for item in scoring_criterias)
    for prompt, response in zip(processed_batch["processed_batch"], cropped_respone_batch):
        rewarding_prompt = f"""
    You are an evaluator. Your task is to grade the response provided by the responder to the user's request based on specific criteria, using a 100-point scale.
    The criteria include:
    {formatted_criteria}

    The responder's answer is formatted as:
    {processed_batch["answer_format"]}

    User's request: "{prompt}"

    Responder's answer: "{response}"

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

async def generate_rewarding_score(rewarding_prompt):
    async def process_api_request(request, index):
        while True:
            try:
                await asyncio.sleep(random.randint(10, 20))
                print(f"Started API request of index: {index}.")
                response = await g4f.ChatCompletion.create_async(
                    model="gpt-4o-mini",
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
    
    tasks = []
    for index, request in enumerate(rewarding_prompt):
        tasks.append(process_api_request(request, index))
    return await asyncio.gather(*tasks, return_exceptions=True)

def model_generate(model, tokenizer, processed_batch, max_sequence_length=1024):
    # Tokenize the input prompt
    inputs = tokenizer(processed_batch['processed_batch'], return_tensors="pt", padding=True)

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generating
    outputs = model.generate(**inputs, max_length=max_sequence_length, output_hidden_states=True, return_dict_in_generate=True)

    # Decode the generated tokens back to text
    model_responses = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]

    # Print out value reference by outputs variable
    print("------------------model_generate------------------")
    referrers = gc.get_referrers(outputs)
    print(f"Number of referrers: {len(referrers)}") 
    # List the referrers
    for ref in referrers:
        print(ref)
    print("--------------------------------------------------")

    # Free memory
    del outputs

    # Free garbage collector
    gc.collect()

    return model_responses

def base_model_generate(base_model, tokenizer, processed_batch):
    # Tokenize the input prompt
    inputs = tokenizer(processed_batch['processed_batch'], return_tensors="pt", padding=True)

    # Move tensors to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generating
    outputs = base_model.generate(**inputs, max_length=1024, output_hidden_states=True, return_dict_in_generate=True)

    # Decode the generated tokens back to text
    base_model_responses = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
    
    # Print out value reference by outputs variable
    print("------------------base_model_generate------------------")
    referrers = gc.get_referrers(outputs)
    print(f"Number of referrers: {len(referrers)}") 
    # List the referrers
    for ref in referrers:
        print(ref)
    print("--------------------------------------------------")

    # Free memory
    del outputs

    # Free garbage collector
    gc.collect()
    return base_model_responses

def split_response_with_prompt(batch):
    splitted_response_batch = []
    for respone in batch:
        temp = respone.split('\nRespone:', 1)[1]
        splitted_response_batch.append(temp)
    return splitted_response_batch

def split_into_chunks(tokenizer, model_response_batch, base_model_respose_batch, length=500):
    chunks_batch = []
    for model_respone, base_model_respone in zip(model_response_batch, base_model_respose_batch):
        # Tokenize the input prompts
        result = []
        pair_respone = [model_respone, base_model_respone]
        inputs = tokenizer(pair_respone, return_tensors="pt", padding=True)

        base_model_respone_input_ids, model_respone_input_ids = torch.split(inputs.input_ids, 1, 0)
        model_attention_mask, base_model_attention_mask = torch.split(inputs.attention_mask, 1, 0)

        # Split input_ids and attention_mask  
        splitted_model_input_ids = torch.split(model_respone_input_ids, length, -1)
        splitted_base_model_input_ids = torch.split(base_model_respone_input_ids, length, -1)
        splitted_model_attention_mask = torch.split(model_attention_mask, length, -1)
        splitted_base_model_attention_mask = torch.split(base_model_attention_mask, length, -1)

        model_chunks = []
        base_model_chunks = []

        for item_input_ids, item_attention_mask in zip(splitted_model_input_ids, splitted_model_attention_mask):
            temp = {"input_ids": item_input_ids, "attention_mask": item_attention_mask}
            model_chunks.append(temp)
        
        for item_input_ids, item_attention_mask in zip(splitted_base_model_input_ids, splitted_base_model_attention_mask):
            temp = {"input_ids": item_input_ids, "attention_mask": item_attention_mask}
            base_model_chunks.append(temp)
        
        # Zip pair model_chunks correspond to base_model_chunks
        result = zip(model_chunks, base_model_chunks)
        
        # Finish processing one sentence, append to chunks_batch
        chunks_batch.append(result)
    return chunks_batch

def split_into_chunks_v2(tokenizer, model_response_batch, length=500):
    chunks_batch = []
    for model_respone in model_response_batch:
        inputs = tokenizer(model_respone, return_tensors="pt", padding=True)

        # Split input_ids and attention_mask  
        splitted_model_input_ids = torch.split(inputs.input_ids, length, -1)
        splitted_model_attention_mask = torch.split(inputs.attention_mask, length, -1)

        model_chunks = []

        for item_input_ids, item_attention_mask in zip(splitted_model_input_ids, splitted_model_attention_mask):
            temp = {"input_ids": item_input_ids, "attention_mask": item_attention_mask}
            model_chunks.append(temp)

        # Finish processing one sentence, append to chunks_batch
        chunks_batch.append(model_chunks)
    return chunks_batch

def model_forward(inputs, model):
    output = model(**inputs)
    return output

def base_model_forward(inputs, base_mode):
    with torch.no_grad():
        output = base_mode(**inputs)
    return output

def exec_and_caculate_average(rewarding_score_text):
    average_rewarding_score = []
    for item in rewarding_score_text:
        # Local variables
        local_vars = {}
        print(item)
        exec(item, {}, local_vars)
        temp = 0
        for reward_item in local_vars['rewarding_score']:
            temp += reward_item['score']
            print(f"temp: {temp}")
        average_rewarding_score.append(torch.tensor(temp / (10 * len(local_vars['rewarding_score'])) + 0.5))
    return average_rewarding_score

def train_prompt(splitted_model_respones, batch):
    output = []
    for i, respone in enumerate(splitted_model_respones):
        prompt = f"""
Develop Blender scripts for animation by analyzing natural scene descriptions, breaking them into individual assets like objects, characters, and props, each with a distinct name and detailed visual description, ensuring no composite sets.
Script: {batch[i]["query"]}

Respone:
{respone}
"""
        output.append(prompt)
    return output

def caculate_KL_diverage_loss(model_chunk_logit, base_model_chunk_logit):
    # Convert logits to probabilities using softmax
    prob1 = F.softmax(model_chunk_logit, dim=-1)
    prob2 = F.softmax(base_model_chunk_logit, dim=-1)

    # Compute KL divergence
    kl_div = F.kl_div(prob1.log(), prob2, reduction='batchmean')
    return kl_div 

def caculate_loss_and_do_gradient_accumulation(tokenizer, model, base_model, batch, scoring_criterias, history):
    # Preprocess data
    processed_batch = step1_preprocess_data(batch=batch)

    # model_generate (require_grad=False)
    # base_model_generate (require_grad=False)
    model_respones = model_generate(model=model, tokenizer=tokenizer, processed_batch=processed_batch)
    # base_model_responses = base_model_generate(base_model=base_model, tokenizer=tokenizer, processed_batch=processed_batch)

    # ----------------------REWARDING SCORE PROCESSING---------------------------
    # crop_response
    cropped_batch = step1_crop_respone(batch=model_respones)

    # craft_rewarding_prompt
    rewarding_prompt = craft_rewarding_prompt(processed_batch=processed_batch, cropped_respone_batch=cropped_batch, scoring_criterias=scoring_criterias)

    # generate_rewarding_score
    rewarding_score_text = asyncio.run(generate_rewarding_score(rewarding_prompt=rewarding_prompt))

    #exec_and_caculate_average
    average_rewarding_score = exec_and_caculate_average(rewarding_score_text=rewarding_score_text)
    print(f"average_rewarding_score: {average_rewarding_score}")
    # -----------------------------------------------------------------------------

    # split_response_with_prompt
    splitted_model_respones = split_response_with_prompt(batch=model_respones)
    # splitted_base_model_respones = split_response_with_prompt(batch=base_model_responses)
    
    prompted_splitted_model_respones = train_prompt(splitted_model_respones=splitted_model_respones, batch=batch)

    # split_into_chunks
    # chunks_batch = split_into_chunks(tokenizer=tokenizer, model_response_batch=splitted_model_respones, base_model_respose_batch=splitted_base_model_respones)
    chunks_batch = split_into_chunks_v2(tokenizer=tokenizer, model_response_batch=prompted_splitted_model_respones)

    # Caculate KL diverage loss, then final loss, then backward
    # Define Beta parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # beta = torch.tensor(0.2).to(device)  
    for rewarding_score, chunks in zip(average_rewarding_score, chunks_batch):
        for model_inputs in chunks:

            # Caculate chunk output of forward 
            model_chunk_output = model_forward(inputs=model_inputs, model=model)
            # base_model_chunk_output = base_model_forward(inputs=base_model_inputs, base_mode=base_model)
            
            # print(f"model_chunk_output: {model_chunk_output.logits} dimension={model_chunk_output.logits.shape} requires_grad={model_chunk_output.logits.requires_grad}")
            # print(f"base_model_chunk_output: {base_model_chunk_output.logits} requires_grad={base_model_chunk_output.logits.requires_grad}")

            # Caculate KL diverage loss
            # kl_loss = caculate_KL_diverage_loss(model_chunk_logit=model_chunk_output.logits, base_model_chunk_logit=base_model_chunk_output.logits)
            
            # print(f"kl_loss: {kl_loss} requires_grad={kl_loss.requires_grad}")

            soft_max = nn.Softmax(dim=-1)

            # Caculate total loss
            total_loss = (-torch.log(rewarding_score) * soft_max(model_chunk_output.logits) / model_chunk_output.logits.shape[1]).sum()  
            # Backward
            if not torch.isnan(total_loss).any():
                total_loss.backward()
                history.append(rewarding_score - 0.5)                
            # Print out value reference by outputs variable
            print("------------------caculate_loss_and_do_gradient_accumulation------------------")
            # print(f"model_inputs: {model_inputs}")
            # print(f"base_model_inputs: {base_model_inputs}")
            print(f"total_loss: {total_loss} requires_grad={total_loss.requires_grad}")
            referrers_model_forward = gc.get_referrers(model_chunk_output)
            # referrers_base_model_forward = gc.get_referrers(base_model_chunk_output)
            print(f"Number of referrers_model_forward: {len(referrers_model_forward)}") 
            # print(f"Number of referrers_base_model_forward: {len(referrers_base_model_forward)}") 
            # List the referrers_model_forward
            for ref in referrers_model_forward:
                print(f"referrers_model_forward: {ref}")
            
            # List the referrers_base_model_forward
            # for ref in referrers_base_model_forward:
                # print(f"referrers_base_model_forward: {ref}")
            print("-------------------------------------------------------------------------------")

            # Free memory
            del model_chunk_output
            # del base_model_chunk_output

            # Free garbage collector
            gc.collect()

def save_model(model):
    diretory=f"/content/drive/MyDrive/adapter_folder"
    model.save_pretrained(diretory)

def train(tokenizer,
          model,
          base_model,
          scoring_criteria,
          train_data_path, 
          num_epoch, 
          batch_size,
          learning_rate,
          start_index,
          end_index,
          shuffle=True,):
    # load train_data
    print("Loading train data...")
    f = open(train_data_path)
    train_data = json.load(f)
    train_data = train_data[start_index:end_index]

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for epoch in range(num_epoch):
        print(f"epoch: {epoch}")

        # Shuffle index data list
        if (shuffle == True):
            index_list = [i for i in range(len(train_data))]
            random.shuffle(index_list)
        else:
            index_list = [i for i in range(len(train_data))]

        num_batch = len(train_data) // batch_size
        
        # Tracking loss history
        history = []

        for i in range(num_batch):
            optimizer.zero_grad()
            
            batch_data = []
            for j in range(i * batch_size, (i + 1)*batch_size):
                batch_data.append(train_data[index_list[j]])
            
            caculate_loss_and_do_gradient_accumulation(tokenizer=tokenizer,
                                                       model=model,
                                                       base_model=base_model,
                                                       batch=batch_data,
                                                       scoring_criterias=scoring_criteria,
                                                       history=history)
            
            optimizer.step()
            print(f"\tepoch: {epoch}, batch: {i}")
        
        print("Saving mode...")
        save_model(model=model)

        print(f"Average Loss: {torch.tensor(history).mean()}")

def test(tokenizer,
        model,
        scoring_criterias,
        test_data_path,
        evaluate_data_path,
        batch_size=1,
        shuffle=True
        ):
    # load test_test
    print("Loading test data...")
    f = open(test_data_path)
    test_data = json.load(f)

    # load evaluate_test
    print("Loading evaluate data...")
    f = open(evaluate_data_path)
    evaluate_data = json.load(f)

    test_data = test_data + evaluate_data

    # Tracking
    answer_format_correctness = []
    scores = []

    num_batch = len(test_data) // batch_size

    # Shuffle index data list
    if (shuffle == True):
        index_list = [i for i in range(len(test_data))]
        random.shuffle(index_list)
    else:
        index_list = [i for i in range(len(test_data))]

    for i in range(num_batch):
        print(f"batch {i}")
        batch_data = []
        for j in range(i * batch_size, (i + 1)*batch_size):
            batch_data.append(test_data[index_list[j]])

            # Preprocess data
            processed_batch = step1_preprocess_data(batch=batch_data)

            # model_generate (require_grad=False)
            model_respones = model_generate(model=model, tokenizer=tokenizer, processed_batch=processed_batch, max_sequence_length=1536)

            # ----------------------REWARDING SCORE PROCESSING---------------------------
            # crop_response
            cropped_batch = step1_crop_respone(batch=model_respones)
            
            # Tracking answer_format_correctness
            for item in cropped_batch:
                if item == "object_list = []":
                    answer_format_correctness.append(0.0)
                else:
                    answer_format_correctness.append(1.0)

            # craft_rewarding_prompt
            rewarding_prompt = craft_rewarding_prompt(processed_batch=processed_batch, cropped_respone_batch=cropped_batch, scoring_criterias=scoring_criterias)

            # generate_rewarding_score
            rewarding_score_text = asyncio.run(generate_rewarding_score(rewarding_prompt=rewarding_prompt))

            #exec_and_caculate_average_score
            for item in rewarding_score_text:
                # Local variables
                local_vars = {}
                print(item)
                exec(item, {}, local_vars)
                temp = 0
                for reward_item in local_vars['rewarding_score']:
                    temp += reward_item['score']
                scores.append(temp / len(local_vars['rewarding_score']))
    
    print(f"Average Score: {torch.tensor(scores).mean()}")
    print(f"Answer Format Correctness Percentage: {torch.tensor(answer_format_correctness).mean() * 100}%")

if __name__ == '__main__':
    # Interface
    print('---------------------------------------------------------------\n')
    print('Please choose setting option \n')
    print('\t1: Fine-tune from scratch\n')
    print('\t2: Resume fine-tune')
    print('\t3: Evaluate')
    setting_option = input('Setting Option (1 to 3): ')
    start_index = input('Start_index (for training): ')
    end_index = input('End_index (for training): ')
    print('---------------------------------------------------------------\n')
    print('Running...')

    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    TRAIN_DATA_PATH = '/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/train_examples.json'
    TEST_DATA_PATH = '/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/test_examples.json'
    EVALUATE_DATA_PATH = '/content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/evaluate_examples.json'

    step1_criteria = [
        {'name': 'Accuracy',
            'description': 'Are the objects identified fully and accurately?'},
        {'name': 'Coverage',
            'description': 'Is any necessary object for the scene missing?'},
        {'name': 'Relevance to Requirements',
            'description': 'Do the objects align with the goals and requirements of the scene?'}
    ]

    # Loading model with setting (Default: Meta-Llama-3-8B-4bit-64rank)
    adapter_folder = f"/content/drive/MyDrive/adapter_folder" 

    # Model and tokenizer IDs
    MODEL_ID = "LoftQ/Meta-Llama-3-8B-4bit-64rank"

    # Loading base model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    if (setting_option == '1'): 
        peft_model = PeftModel.from_pretrained(
                base_model,
                MODEL_ID,
                subfolder="loftq_init",
                is_trainable=True)
        
        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            model_max_length=1536,
            padding_side="right",
            use_fast=False)
    
        # Adding special token to tokenizer
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=peft_model,
            base_model=base_model
        )
    elif (setting_option == '2'):
        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            model_max_length=1536,
            padding_side="right",
            use_fast=False)
    
        # Adding special token to tokenizer
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
        smart_tokenizer_and_embedding_resize_base_model(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            base_model=base_model
        )

        peft_model = PeftModel.from_pretrained(
                base_model,
                adapter_folder,
                is_trainable=True)
    elif (setting_option == '3'):
        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            model_max_length=1536,
            padding_side="right",
            use_fast=False)
    
        # Adding special token to tokenizer
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
        smart_tokenizer_and_embedding_resize_base_model(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            base_model=base_model
        )

        peft_model = PeftModel.from_pretrained(
                base_model,
                adapter_folder,
                is_trainable=True)
    
    if setting_option == '1' or setting_option == '2':
        train(tokenizer=tokenizer,
            model=peft_model,
            base_model=base_model,
            scoring_criteria=step1_criteria,
            train_data_path=TRAIN_DATA_PATH,
            num_epoch=1,
            batch_size=1,
            learning_rate=3e-7,
            shuffle=True,
            start_index=int(start_index),
            end_index=int(end_index))
    if setting_option == '3':
        test(tokenizer=tokenizer,
             model=peft_model,
             scoring_criterias=step1_criteria,
             test_data_path=TEST_DATA_PATH,
             evaluate_data_path=EVALUATE_DATA_PATH
             )
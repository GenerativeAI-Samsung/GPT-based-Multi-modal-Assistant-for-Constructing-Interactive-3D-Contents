import json

from LanguageModel import ModifyModel

if __name__ == '__main__':    
    MODEL_ID = "apple/OpenELM-270M"
    TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
    adapter_layer1='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step1_LoRA_ModifyPart', 
    adapter_layer2='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step2_LoRA_ModifyPart', 
    adapter_layer3='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step3_LoRA_ModifyPart',
    adapter_layer4='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step4_LoRA_ModifyPart',
    adapter_layer5='./content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Step5_LoRA_ModifyPart'
    classify_request = './content/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/Classify_Modify_LoRA'
    print("------------------------------------------------------")
    print("Initialize OpenELM-270M...")
    modify_model = ModifyModel(MODEL_ID=MODEL_ID, 
                                        TOKENIZER_ID=TOKENIZER_ID,
                                        adapter_layer1=adapter_layer1,
                                        adapter_layer2=adapter_layer2,
                                        adapter_layer3=adapter_layer3,
                                        adapter_layer4=adapter_layer4,
                                        adapter_layer5=adapter_layer5,
                                        classify=classify_request)
    print("Done!")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Loading initial request...")
    with open('user_interact_result.json', 'r') as openfile:
        original_prompt = json.load(openfile)
    original_prompt = original_prompt["user_interact_result"][0]
    print("------------------------------------------------------")        
    print("Loading previous answer...")
    print("     Loading step1...")
    with open('step1_respone.json', 'r') as openfile:
        step1_previous = json.load(openfile)    
    print("     Loading step2...")
    with open('step2_respone.json', 'r') as openfile:
        step2_previous = json.load(openfile)    
    print("     Loading step3...")
    with open('step3_respone.json', 'r') as openfile:
        step3_previous = json.load(openfile)    
    print("------------------------------------------------------")
    request = input("Please input your request: ")
    print("------------------------------------------------------\n")
    print("Classify your response...")
    modify_step = modify_model.classify_generate(original_prompt=original_prompt,
                                                 modify_prompt=request)
    print("Done!")
    print("------------------------------------------------------")
    if (modify_step == 1):
        print("\n------------------------------------------------------")
        print("Identify key objects")
        step1_respone = [modify_step.step1_generate(original_prompt=original_prompt,
                                                    original_object_list=step1_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step1_respone, indent=4)
        with open("step1_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step1_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step1_respone.json', 'r') as openfile:
            step1_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step1_respone[0])

        print("\n------------------------------------------------------")
        print("Determine the initial position of the objects")
        step2_respone = [modify_step.step2_generate(original_prompt=original_prompt,
                                                    original_init_pos=step2_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step2_respone, indent=4)
        with open("step2_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step2_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step2_respone.json', 'r') as openfile:
            step2_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step2_respone[0])   

        print("\n------------------------------------------------------")
        print("Generates actions and movements of objects")
        step3_respone = [modify_step.step3_generate(original_prompt=original_prompt,
                                                    original_movements=step3_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 
    elif (modify_step == 2):
        print("\n------------------------------------------------------")
        print("Determine the initial position of the objects")
        step2_respone = [modify_step.step2_generate(original_prompt=original_prompt,
                                                    original_init_pos=step2_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step2_respone, indent=4)
        with open("step2_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step2_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step2_respone.json', 'r') as openfile:
            step2_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step2_respone[0])   

        print("\n------------------------------------------------------")
        print("Generates actions and movements of objects")
        step3_respone = [modify_step.step3_generate(original_prompt=original_prompt,
                                                    original_movements=step3_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 
    elif (modify_step == 3):
        print("\n------------------------------------------------------")
        print("Generates actions and movements of objects")
        step3_respone = [modify_step.step3_generate(original_prompt=original_prompt,
                                                    original_movements=step3_previous,
                                                    modify_prompt=request)]
        print("Done!")
        print("------------------------------------------------------")

        json_object = json.dumps(step3_respone, indent=4)
        with open("step3_respone.json", "w") as outfile:
            outfile.write(json_object)
        print("You should check respone in step3_respone.json to make sure that response reliable and executable!")
        control = None
        while (control != 'continue'):
            control = input('Press "continue" if done: ')
        
        with open('step3_respone.json', 'r') as openfile:
            step3_respone = json.load(openfile)
        print("------------------------------------------------------")
        exec(step3_respone[0]) 

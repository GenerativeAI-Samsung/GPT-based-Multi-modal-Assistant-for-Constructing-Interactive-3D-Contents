from RAG import RAG_module
from LanguageModel import ModifyPart
import json
import torch
import asyncio

def reCheck(obj, path_to_obj):
    json_obj = json.dumps(obj, indent=4)
    with open(path_to_obj, "w") as outfile:
        outfile.write(json_obj)

    temp_var = None    
    while (temp_var != 'Okay!'):
        temp_var = input(f"Re-checking {path_to_obj} to make sure thing right, then press 'Okay!' to continue:")

if __name__ == '__main__':
    env = {}

    print("Loading scene...")
    with open('/content/script.json', 'r') as openfile:
        script = json.load(openfile)
    with open('/content/step1.json', 'r') as openfile:
        step1_prev = json.load(openfile)    
    with open('/content/step2.json', 'r') as openfile:
        step2_prev = json.load(openfile)
    with open('/content/step3.json', 'r') as openfile:
        step3_prev = json.load(openfile)
    print("Done!\n")
    
    print("Loading ModifyPart model...")
    modifyPart = ModifyPart()
    print("Done!\n")

    request = input("What do you want to change? - ")

    print("Classify to step need to change...")
    modify_step = modifyPart.classify_generate(ori_req=script,
                                               modify_res=request)
    modify_step = int(modify_step)
    print("Step need to modify is: ", modify_step, "\n")

    if (modify_step == 1):
        print("Running Step1...")
        print(script)
        res = ModifyPart.step1_modify(ori_req=script,
                                      ori_object_list=step1_prev,
                                      modify_res=request)
        exec(res, env)
        reCheck(env["object_list"], "/content/step1_modify.json")
        print("Done!")
        print("--------------------------------------------------------")

        print("Running Step2...")
        res = ModifyPart.step2_modify(ori_req=script,
                                      modified_object_list=env["object_list"],
                                      ori_init_pos=step2_prev,
                                      modify_res=request)
        exec(res, env)
        reCheck(env["init_pos"], "/content/step2_modify.json")
        print("Done!\n")
        print("--------------------------------------------------------")

        print("Running Step3...")
        res = ModifyPart.step3_modify(ori_req=script,
                                      ori_movs=step3_prev,
                                      modify_res=request)
        exec(res, env)
        reCheck(env["movements"], "/content/step3_modify.json")
        print("Done!\n")
        print("--------------------------------------------------------")
    elif (modify_step == 2):
        print("Running Step3...")
        res = ModifyPart.step3_modify(ori_req=script,
                                      ori_movs=step3_prev,
                                      modify_res=request)
        exec(res, env)
        reCheck(env["movements"], "/content/step3_modify.json")
        print("Done!\n")
        print("--------------------------------------------------------")
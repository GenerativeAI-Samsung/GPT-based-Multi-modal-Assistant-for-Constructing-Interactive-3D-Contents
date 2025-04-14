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
    request = input("What do you want to change? - ")

    
import json
from LanguageModel import ScriptToScene

def reCheck(obj, path_to_obj):
    json_obj = json.dumps(obj, indent=4)
    with open(path_to_obj, "w") as outfile:
        outfile.write(json_obj)

    temp_var = None    
    while (temp_var != 'Okay!'):
        temp_var = input(f"Re-checking {path_to_obj} to make sure thing right, then press 'Okay!' to continue:")

if __name__ == '__main__':
    env = {}

    print("Loading script...")
    with open('/content/script.json', 'r') as openfile:
        script = json.load(openfile)
    script = script["script"]
    print("Done!\n")

    print("Loading ScriptToScene...")
    scriptToScene = ScriptToScene()
    print("Done!\n")

    print("Running Step1...")
    res = scriptToScene.step1(request=script)
    exec(res, env)
    reCheck(env["object_list"], "step1.json")
    print("Done!")
    print("--------------------------------------------------------")

    print("Running Step2...")
    res = scriptToScene.step2(request=script, 
                              object_list=env["object_list"])
    exec(res, env)
    reCheck(env["init_pos"], "step2.json")
    print("Done!\n")
    print("--------------------------------------------------------")

    print("Running Step3...")
    res = scriptToScene.step3(request=script, 
                              object_list=env["object_list"],
                              init_pos=env["init_pos"])
    exec(res, env)
    reCheck(env["movements"], "step3.json")
    print("Done!\n")
    print("--------------------------------------------------------")

    print("Running Step4...")
    res = scriptToScene.step4(request=script)
    exec(res, env)
    reCheck(env["object_evironment_list"], "step4.json")
    print("Done!\n")
    print("--------------------------------------------------------")

    print("Running Step5...")
    res = scriptToScene.step5(request=script,
                              object_evironment_list=env["object_evironment_list"])
    exec(res, env)
    reCheck(env["init_enviroment_pos"], "step5.json")
    print("Done!\n")
    print("--------------------------------------------------------")
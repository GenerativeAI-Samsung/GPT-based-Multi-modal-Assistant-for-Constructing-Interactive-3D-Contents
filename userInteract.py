from RAG import RAG_module
from LanguageModel import deepseek_generate, VisionLangugeModel
import json
import torch

def reCheck(obj, path_to_obj):
    json_obj = json.dumps(obj, indent=4)
    with open(path_to_obj, "w") as outfile:
        outfile.write(json_obj)
    
    temp_var = None
    while (temp_var != 'Okay!'):
        temp_var = print(f"Re-checking {path_to_obj} to make sure thing right, then press 'Okay!' to continue:")

if __name__ == '__main__':
    request = input("What can I help you? - ")
    ext_txts = input("Please provide some addition documents:")

    ext_imgs = []

    print("Please provide some additional images")
    temp_var = None
    while (temp_var != "That's all!"):
        temp_img_path = input("\tPlease provide Image's path: ")
        temp_img_desc = input("\tPlease provide Image's description: ")
        ext_imgs.append({"image_path": temp_img_path, 
               "image_description": temp_img_desc,
               "image_questions": None,
               "questions_response": None})
    temp_var = input("Do you want to add any image more? - ")

    doc_aug = [ext_txts]
    print("\nDeepSeekv3 is asking questions about your images...")
    ques = []
    for i, img in enumerate(ext_imgs):
        prompt = f"""
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
However, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the description about the image to extract the needed information to fulfill the user's request.

Description: "{img["image_description"]}"

After getting the answers, format them as follows:
external_images[{i}]["image_questions"] = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
"""
        res = await deepseek_generate(prompt=[prompt])
        ques.append({"index": i, "prompt": prompt, "respone": res})
    print("DONE!\n")
    
    reCheck(ques, "question_asked.json")
    for item in ques:
        exec(item['response'])

    print("Initialize Vison Language Model (MC-LLaVA-3b)...")
    vision_lm = VisionLangugeModel()
    print("Done!\n")

    print("Vision Language model is answering question about image...")

    ans = []
    for i, img in enumerate(ext_imgs):
        temp_lst = []
        for ques in img["image_questions"]:
            prompt = f"""
Based on the image and the description of the picture below, please answer the following questions related to the image
Some question may not relevant to the image and description. In that case, you should answer "Unknown"
Description of the image: {img["image_description"]}
Question: {ques}
"""
            res = vision_lm.process(image_path=img["image_path"],query=prompt)
            temp_lst.append({"image_index": i, "question": ques, "respone": res})
        ans.append(temp_lst)
    reCheck(ans, "answers.json")
    for i, img in enumerate(ext_imgs):
        img["questions_response"] = [item["response"] for item in ans[i]]
    for img in ext_imgs:
        doc_aug.extend(res for res in img["questions_response"])
    
    print(f"Document Augmented: {doc_aug}\n")

    print("Intizalize RAG...")
    retrival_module = RAG_module()
    retrival_module.initialize_embedding_database(text=doc_aug)
    print("Done!\n")

    print("Start User Interact Interface...")
    res = ""
    while (request != "That's all!"):
        retrival_content = retrival_module.find_top_k_embedding(query=request, k=20)
        retrival_content = ''.join((item + "\n") for item in retrival_content)

        prompt = f"""
Based on user request, describe a 3D scene in a single, continuous paragraph. The description must include:
    1.Location: Where the scene takes place.
    2.Initial Placement of Objects: The starting position of all necessary objects in the scene.
    3.Object Movements: Movements taken from the following list, ensuring each movement includes the required parameters:
        1. zigZag: Parameters include starting and ending coordinates.
        2. standStill: No parameters.
        3. runStraight: Parameters include starting and ending coordinates.
        4. ellipse: Parameters include starting coordinates, center coordinates, major axis length, and minor axis length.
        5. jump: Parameters include starting coordinates, peak coordinates, and ending coordinates.
        6. Linear Bézier Curve: Parameters include two points (P0 and P1).
        7. Quadratic Bézier Curve: Parameters include three points (P0, P1, P2).
        8. Cubic Bézier Curve: Parameters include four points (P0, P1, P2, P3).
    4. Frame Range: The starting and ending frame for each object's motion.
Ensure the description flows naturally and avoids numbered sections or headers. It must integrate all required details seamlessly.

Once the output is determined, it must be presented in the following format:   output=["your answer"]
Avoid using normal text; format your response strictly as specified above.

User Request: {request}
"""
        prompt += f"""
Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:
{retrival_content}
"""
        print("Responsing...")
        res = await deepseek_generate(prompt=[prompt])
        print(f"DeepSeek v3's Response: \n{res}")
        request = input("Do you have any further modification? (press 'That's all!' if no):")

    json_obj = json.dumps({"script": res})
    with open("/content/script.json", "w") as outfile:
        outfile.write(json_obj)
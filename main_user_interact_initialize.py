from RAG import RAG_module
from LanguageModel import VisionLangugeModel
import json

if __name__ == '__main__':

# Kịch bản thử nghiệm 
# Đầu vào hệ thống gồm 3 phần:
#   - User Query: Chứa phần nội dung tương tác chính của người dùng tới mô hình ngôn ngữ
#   - External Text Resource: Chứa phần nội dung text thông tin người dùng muốn mô hình dựa vào để trả lời
#   - External Image Resource: Chứa hình ảnh, kèm discription, mà người dùng muốn thêm vào 
# -----------------------------------------------------------------------------
    # User Query
    input = 'Create 3D scene for this text: "A girl plays with her dog in the garden"'

    # External Text Resource
    external_texts = """
The circular path that wraps around the lawn can be used for cycling around, running, and for the adults, walking around to admire the garden.
A sandpit or blowup plunge pool could be placed on the patio, a swing ball (if those still exist) or mini-trampoline on the lawn and suddenly the nicely designed adult garden is as welcoming for children as it is for adults without it having to look like a children's playground.
Dens can be made from cardboard boxes joined together, old chairs and tarpaulins and whatever else you've got stuffed away unused in the garage!
Make sure you put tough plants around lawn areas which can take having balls and small children land on them. So lavenders with their woody stems will fair much better than soft fleshy plants like hostas and herbaceous peonies.
"""

    # External Image Resource 
    external_images = [{"image_path": "/content/Image1.jpg", 
               "image_description": "The garden should look like this",
               "image_questions": None,
               "questions_response": None},
              {"image_path": "/content/Image2.jpg", 
               "image_description": "I want the girl character look like this",
               "image_questions": None,
               "questions_response": None},
              {"image_path": "/content/Image3.jpeg", 
               "image_description": "And the dog should look like this",
               "image_questions": None,
               "questions_response": None}] 
# -----------------------------------------------------------------------------

# Đối với External Text Resource, trựa tiếp đưa vào list các documents_augement
    documents_augement = [external_texts]
    
# Đối với External Image Resource, mô hình ngôn ngữ chính sẽ đặt một số các câu hỏi liên quan đến các hình ảnh dựa vào: User query, Images_discription

    # Prompt để mô hình ngôn ngữ chính đặt câu hỏi
    for i, image in enumerate(external_images):
        prompt = f"""
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
However, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the user's request and the description about the image to extract the needed information to fulfill the user's request.

Request: "{input}"

Description: "{image["image_description"]}"

After getting the answers, format them as follows:
external_images[{i}]["image_questions"] = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
"""
        
    question1s = """
external_images[0]["image_questions"] = [
    "What are the main features of the garden in the picture? (e.g., flowers, trees, lawn, etc.)",
    "What type of dog is in the picture? (e.g., breed, size, color)",
    "What is the girl wearing in the picture? (e.g., dress, shorts, color, style)",
    "What is the overall color palette of the garden? (e.g., green, colorful, muted tones)",
    "Are there any notable objects or elements in the garden? (e.g., a bench, fountain, specific type of plant)",
    "What time of day does it appear to be in the picture? (e.g., morning, afternoon, evening)",
    "Is there any specific weather or lighting in the garden? (e.g., sunny, cloudy, shadowy)",
    "What is the general layout of the garden? (e.g., paths, flower beds, open space)",
    "Are there any additional characters or animals in the garden besides the girl and the dog?",
    "What is the background scenery beyond the garden? (e.g., fence, house, forest)"
]
"""

    question2s = """
external_images[1]["image_questions"] = [
    "What is the girl wearing in the picture? (e.g., dress, shorts, color, style)",
    "What is the hairstyle of the girl in the picture? (e.g., long, short, braided, color)",
    "What is the girl's physical appearance in the picture? (e.g., height, build, facial features)",
    "What is the age or approximate age of the girl in the picture?",
    "Is the girl wearing any accessories in the picture? (e.g., hat, glasses, jewelry)",
    "What is the girl's pose or activity in the picture? (e.g., standing, sitting, running)",
    "What is the overall color palette or style of the girl's clothing in the picture? (e.g., bright, pastel, casual, formal)",
    "Are there any notable expressions or emotions visible on the girl's face in the picture?",
    "Does the girl have any distinguishing features in the picture? (e.g., freckles, scars, birthmarks)",
    "Is the girl holding or interacting with any objects in the picture? (e.g., toys, books, sports equipment)"
]
"""

    question3s = """
external_images[2]["image_questions"] = [
    "What breed is the dog in the picture?",
    "What is the size of the dog in the picture? (e.g., small, medium, large)",
    "What is the color and pattern of the dog's fur in the picture?",
    "What is the dog's pose or activity in the picture? (e.g., sitting, running, playing)",
    "Does the dog have any accessories in the picture? (e.g., collar, leash, bandana)",
    "What is the dog's physical appearance in the picture? (e.g., ear shape, tail length, body build)",
    "Is there any notable expression or emotion visible on the dog's face in the picture?",
    "Are there any distinguishing features on the dog in the picture? (e.g., spots, scars, unique markings)",
    "What is the overall condition of the dog's fur in the picture? (e.g., well-groomed, messy, curly, straight)",
    "Is the dog interacting with any objects or characters in the picture? (e.g., toys, the girl, other animals)"
]
"""
    exec(question1s)
    exec(question2s)
    exec(question3s)

    # Các câu hỏi vừa xong tiếp đến sẽ được mô hình Vision_LM trả lời
    Vision_LM = VisionLangugeModel()

    for i, image in enumerate(external_images):
        for question in image["image_questions"]:
            # Prompt cho mô hình Vision Language trả lời, dựa vào: Hình ảnh, Image description, Image Question
            prompt = f"""
Based on the image and the description of the picture below, please answer the following questions related to the image
Description of the image: {image["image_description"]}
Question: {question}
"""
            respone = Vision_LM.process(image_path=image["image_path"],query=prompt)
            print(respone)

    external_images[0]["questions_response"] = [
    "Several plants and flowers are growing in a garden. The garden has a green lawn, red brick path, and a wooden fence. There are also many colorful pots and flowers in the garden.",
    "We have several pictures, so I cannot see or describe each dog.",
    "She is wearing a dress and a pair of high heels.",
    "The overall color palette of the garden is green and colorful. The green lawn is dotted with colorful flowers, including pink, purple, and yellow flowers.",
    "The garden has some notable elements, including several different pots with plants in various sizes and positions, a water feature, and a bench where one could sit and enjoy the view.",
    "It appears to be a sunny afternoon because there is direct sunlight shining on the plants.",
    "Sunlight falls on the garden from the top, casting shadows on the grass and plants.",
    "The general layout of the garden in the image includes a mixture of open space, flower beds, and pathways. There are four planter boxes containing different plants, and some bushes and trees also contribute to the green scenery.",
    "No",
    "The fence on the left.",
    ]

    external_images[1]["questions_response"] = [
    "The girl is wearing a yellow sweater.",
    "The girl has her hair in a low ponytail.",
    "A young blonde girl with blue eyes is standing with a big smile. She has a plaid sweater on, a white cardigan sweater over it, and a white shirt underneath. She has a pink headband on and is standing in front of a fireplace.",
    "The girl in the picture appears to be young, and although it is difficult to pinpoint an exact age, she could be a little girl or a pre-teen.",
    "No",
    "The girl is posing for the picture.",
    "The girl's clothing in the picture is mostly bright and colorful.",
    "Yes",
    "The girl has a smile and a smiley face on her t-shirt and is smiling brightly at the camera.",
    "No"
    ]

    external_images[2]["questions_response"] = [
    "The dog in the picture is an Australian Shepherd.",
    "The dog in the picture is medium-sized.",
    "The dog has a black, tan, and white coat with a brown or reddish-brown tip on its muzzle.",
    "The dog is walking and is not on a leash.",
    "Yes, the dog has a red dog bandana on its neck.",
    "The dog in the picture is of medium build, with a brown, black, and white coat and a long tail. Its ears are long and floppy, and it has a happy expression on its face with its mouth slightly open and teeth showing. The dog is standing on the lush green grass, with its tail wagging at the end of its tail, showing its relaxed and content demeanor.",
    "Yes",
    "Yes, the dog in the picture has some distinguishing features: it has brown, black, and white fur; its tongue is hanging out in a smile-like shape; its face has a unique coloration; and its hair is long.",
    "The dog's fur appears to be well-groomed, with some curly elements and an overall neat and tidy appearance.",
    "No, the dog in the image is not interacting with any objects or characters such as toys or the girl. The dog is just standing in a grassy field, alone."
    ]

    # Các câu trả lời của mô hình ngôn ngữ sẽ được đưa vào list các documents_augement để về sau mô hình ngôn ngữ truy vấn
    for image in external_images:
        documents_augement.extend(image["questions_response"])

    # Khởi tạo RAG module
    Retrieval_module = RAG_module()
    Retrieval_module.initalize_embedding_database(text=documents_augement)

    # Sau khi khởi tạo, lưu lại các embedding vector thành file .json
    json_object = json.dumps(Retrieval_module.embedding_dicts)

    with open("/content/embedding_dicts.json", "w") as outfile:
        outfile.write(json_object)

    # Thực hiện việc truy xuất thông tin từ RAG module (Top 10)
    output_RAG = RAG_module.find_top_k_embedding(query=input, k=10)

    output_RAG = ['toys or the girl. The dog is just standing in a grassy field, alone.', 
                  'No, the dog in the image is not interacting with any objects or characters such as toys or the girl.', 
                  'We have several pictures, so I cannot see or describe each dog.', 
                  "The girl's clothing in the picture is mostly bright and colorful.", 
                  'The dog in the picture is medium-sized.', 
                  'The girl is wearing a yellow sweater.', 
                  'She is wearing a dress and a pair of high heels.', 
                  'The girl in the picture appears to be young, and although it is difficult to pinpoint an exact age,', 
                  "to look like a children's playground.", 
                  'an exact age, she could be a little girl or a pre-teen.']


    # Sau đó, prompt để mô hình ngôn ngữ trả lời dựa trên thông tin được lựa từ output_RAG
    prompt = """
You are a friendly assistant. Your task is to interact with the user to create a script that meets the user's requirements.

User input: {input} 

Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:
{output_RAG}

Your answer should contain natural language only
"""

    # respone sẽ là đầu vào cho file "main_3D_scene_create.py"
    respone = """
To create a 3D scene for the text "A girl plays with her dog in the garden," we can use the following details:

    The Setting:
        The scene takes place in a garden, which includes a grassy field.
        Consider adding elements to make it look like a children's playground.

    The Girl:
        She is young, possibly a little girl or a pre-teen.
        She is wearing a yellow sweater, a dress, and a pair of high heels.
        Her clothing is mostly bright and colorful.

    The Dog:
        The dog is medium-sized.
        The dog can be portrayed standing or interacting with the girl.

    Interaction:
        The girl is playing with the dog, so they should be interacting with each other in a playful manner.

Based on these details, you can imagine a vibrant and joyful scene. 
The girl, dressed in her colorful outfit, is happily playing with her medium-sized dog in a lush, green garden. 
The garden might include elements of a playground to enhance the playful atmosphere.
"""
from RAG import RAG_module
from LanguageModel import VisionLangugeModel

if __name__ == '__main__':

    input = 'a girl play with her dog in the garden'
    images = ["path/to/image1", "path/to/image2", ...]
    addition_text = 'addition text'
    query = "query"

    documents_augement = [addition_text]
    
    # To the main Language Model

    # Language Model ask question about the Images
    question_query = """
You are an assistant that helps answer user requests.
The user provides you with some pictures, and you have to respond to their request based on those pictures.
dHowever, you do not have direct access to the pictures. The only way to approach them is by asking questions related to the pictures to gather the necessary information.
Your task is to create questions based on the user's request below to extract the needed information to fulfill the user's request.

Request: "{input}"

After getting the answers, format them as follows:
questions = [question1, question2, ...]

Avoid using normal text; format your response strictly as specified above.
"""

    output = """ 
    questions = ["What is the girl wearing?", "What breed is the dog?", "What color is the dog?", "What is the girl doing while playing with the dog?", "Is the girl holding any toys or objects?", "Is the dog doing any tricks or specific actions?", "What is the condition of the garden?", "Are there any other people or animals in the garden?", "What time of day is it in the picture?", "Is there any notable background or scenery in the garden?"]
"""
    exec(output)

    Vision_LM = VisionLangugeModel()

    for image in images:
        for question in questions:
            respone = Vision_LM.process(image_path=image, query=question)
            documents_augement.append(respone)
    

    # Initialize embedding database
    RAG_module = RAG_module
    for document in documents_augement:
        RAG_module.initalize_embedding_database(text=document)
    

    output_RAG = RAG_module.find_top_k_embedding(query=query, k=10)


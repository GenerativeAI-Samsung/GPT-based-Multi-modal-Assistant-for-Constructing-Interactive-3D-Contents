import time
import asyncio
import g4f
import json
import random

async def process_api_request(request, index):
    while True:
        try:
            await asyncio.sleep(random.randint(10, 20))
            print(f"Started API request of index: {index}.")
            response = await g4f.ChatCompletion.create_async(
                model="gpt-4o",
                messages=[{"role": "user", "content": request}],
            )
            if len(response) == 0:
                continue
            print(f"Completed API request of index: {index}")
            return response

        except Exception as e:
            print(f"Request of index {index} - Error: {str(e)}")
            await asyncio.sleep(10)

async def run_concurrent_requests(concurrent_requests):
    #async with Client() as session:
        tasks = []
        for index, request in enumerate(concurrent_requests):
            tasks.append(process_api_request(request, index))
        return await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    start = time.time()
    print("Loading train data...")
    f = open('/content/train_examples.json')
    train_data = json.load(f)

    print("Loading test data...")
    f = open('/content/test_examples.json')
    test_data = json.load(f)

    print("Loading evaluate data...")
    f = open('/content/evaluate_examples.json')
    evaluate_data = json.load(f)

    # Train prompts
    train_requests = [f"""
You are a friendly assistant. Your task is to interact with the user to create a script that meets the user's requirements.

User input: Create 3D scene for this text: "{item['query']}"

Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:

Your answer should contain natural language only
""" for item in train_data]

    # Test prompts
    test_requests = [f"""
You are a friendly assistant. Your task is to interact with the user to create a script that meets the user's requirements.

User input: Create 3D scene for this text: "{item['query']}"

Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:

Your answer should contain natural language only
""" for item in test_data]

    # Evaluate prompts
    evaluate_requests = [f"""
You are a friendly assistant. Your task is to interact with the user to create a script that meets the user's requirements.

User input: Create 3D scene for this text: "{item['query']}"

Additionally, there is some supplementary information that will help you respond more accurately to the user's needs:

Your answer should contain natural language only
""" for item in evaluate_data]

    print("START GET RESPONSE FOR TRAIN DATA... ")
    responses = asyncio.run(run_concurrent_requests(concurrent_requests=train_requests))
    end = time.time()
    print(f"\tTime elapsed: {end - start}")

    for i, (resp, item) in enumerate(zip(responses, train_data)):
         train_data[i]["respone"] = resp

    print("saving...")
    json_object = json.dumps(train_data, indent=4)
    with open('/content/train_examples.json', "w") as outfile:
        outfile.write(json_object)

    print("FINISH!")
    print("-------------------------------------")
    print("START GET RESPONSE FOR TEST DATA... ")
    responses = asyncio.run(run_concurrent_requests(concurrent_requests=test_requests))
    end = time.time()
    print(f"\tTime elapsed: {end - start}")

    for i, (resp, item) in enumerate(zip(responses, test_data)):
         test_data[i]["respone"] = resp

    print("saving...")
    json_object = json.dumps(test_data, indent=4)
    with open('/content/test_examples.json', "w") as outfile:
        outfile.write(json_object)

    print("FINISH!\n")
    print("-------------------------------------")
    print("START GET RESPONSE FOR EVALUATE DATA... ")
    responses = asyncio.run(run_concurrent_requests(concurrent_requests=evaluate_data))
    end = time.time()
    print(f"\tTime elapsed: {end - start}")

    for i, (resp, item) in enumerate(zip(responses, evaluate_data)):
         evaluate_data[i]["respone"] = resp

    print("saving...")
    json_object = json.dumps(evaluate_data, indent=4)
    with open('/content/evaluate_examples.json', "w") as outfile:
        outfile.write(json_object)

    print("FINISH!\n")
    print("-------------------------------------")

from tqdm import tqdm
import multiprocessing
from openai import OpenAI
from argparse import ArgumentParser
import logging
import os
import jsonlines

def set_gpt():
    api_key = "sk-SPmgatijZMoa4g4WE9Ea5c9d0a124d4dBb43418b88C0Ae69"
    api_base = "https://api.xi-ai.cn/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client

def get_prompt(mode, item):
    prompt_open = """
    I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

    Please evaluate the response on a scale of 1 to 5:
    1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
    2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
    3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
    4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
    5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

    Below are the transcription of user’s instruction and models’ response:
    ### [Instruction]: {question}
    ### [Response]: {answer}

    After evaluating, please output the score only without anything else.
    You don’t need to provide any explanations.
    """.strip()

    prompt_qa = """
    ### Question
    {question}

    ### Reference answer
    {reference}

    ### Candidate answer
    {answer}

    Is the candidate answer correct based on the question and reference answer? 
    Please only output a single "Yes" or "No". Do not output anything else.
    """.strip()

    prompt_contrast = """
    I need your help to compare the performance of two models in the speech interaction scenario. The two models will receive the same speech input from the user, which they need to understand and respond to with a speech output.
    Your task is to compare the two models’ responses based on the provided user input transcription [Question] and the two models’ output transcription [Answer_1] and [Answer_2].
    You need to evaluate the outputs of the two models comprehensively based on their Relevance, Accuracy, Clarity, and Completeness.

    Please provide a score between -2 and 2 based on the following criteria:
    -2 point: [Answer_1] is significantly worse than [Answer_2].
    -1 points: [Answer_1] is slightly worse than [Answer_2].
    0 points: [Answer_1] and [Answer_2] are equally good or bad.
    1 points: [Answer_1] is slightly better than [Answer_2].
    2 points: [Answer_1] is significantly better than [Answer_2].

    Below are the transcription of user’s instruction and models’ response:
    ### Question
    {question}

    ### Answer_1
    {answer_1}

    ### Answer_2
    {answer_2}

    After evaluating, please output the score only without anything else.
    You don’t need to provide any explanations.
    """.strip()

    if mode == "open":
        return prompt_open.replace("{question}", item["question"]).replace("{answer}", item["answer"])
    elif mode == "qa":
        return prompt_qa.replace("{question}", item["question"]).replace("{reference}", item["reference"]).replace("{answer}", item["answer"])
    elif mode == "contrast":
        return prompt_contrast.replace("{question}", item["question"]).replace("{answer_1}", item["answer_1"]).replace("{answer_2}", item["answer_2"])

def mark(prompt, client):
    scores = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}],
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    temperature=0.5, top_p=0.95, n=3
                    )    
    return scores

def eval(args, client):
    if args.mode == "qa":
        output_file = os.path.join(args.output_dir, 'result_qa.jsonl')
        sum_score = 0
        with open(args.question, 'r') as f:
            length = sum([1 for _ in f])
        with open(args.question, 'r') as qt, open(args.answer, 'r') as pt, jsonlines.open(output_file, mode='w') as ot:
            for question, answer in tqdm(zip(qt, pt), total=length):
                item = {"question": question, "answer": answer}
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = [choice.message.content for choice in scores.choices]
                ot.write(item)
                sum_score += sum([(1 if i == "Yes" else 0) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score / length})
        # save results
        logging.info(f"saving result to {output_file}")
    
    elif args.mode == "open":
        output_file = os.path.join(args.output_dir, 'result_open.jsonl')
        sum_score = 0
        with open(args.question, 'r') as f:
            length = sum([1 for _ in f])
        with open(args.question, 'r') as qt, open(args.answer, 'r') as pt, open(args.reference, 'r') as gt, jsonlines.open(output_file, mode='w') as ot:
            for question, answer, reference in tqdm(zip(qt, pt, gt), total=length):
                item = {"question": question, "answer": answer, "reference": reference}
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = [choice.message.content for choice in scores.choices]
                ot.write(item)
                sum_score += sum([int(i) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score / length})
        # save results
        logging.info(f"saving result to {output_file}")

    elif args.mode == "contrast":
        output_file = os.path.join(args.output_dir, 'result_contrast.jsonl')
        sum_score = 0
        with open(args.question, 'r') as f:
            length = sum([1 for _ in f])
        with open(args.question, 'r') as qt, open(args.answer, 'r') as pt, open(args.answer_contrast, 'r') as ct, jsonlines.open(output_file, mode='w') as ot:
            for question, answer, answer_contrast in tqdm(zip(qt, pt, ct), total=length):
                item = {"question": question, "answer_1": answer, "answer_2": answer_contrast}
                prompt = get_prompt(args.mode, item)
                scores = mark(prompt, client)
                item["score"] = [choice.message.content for choice in scores.choices]
                ot.write(item)
                sum_score += sum([int(i) for i in item["score"]]) / len(item["score"])
            ot.write({"final_score": sum_score / length})
        # save results
        logging.info(f"saving result to {output_file}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=["qa", "open", "contrast"])
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--answer', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--reference', type=str, required=False)
    parser.add_argument('--answer_contrast', type=str, required=False)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
		level=logging.INFO, 
		format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)

    client = set_gpt()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # evaluate data
    logging.info("<========start to evaluate data========>")
    eval(args, client)

if __name__ == '__main__':
    main()
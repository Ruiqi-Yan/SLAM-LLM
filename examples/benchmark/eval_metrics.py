import jiwer
from tqdm import tqdm
import multiprocessing
from argparse import ArgumentParser
import logging
import os
import jsonlines
import inflect

def convert_numbers_to_words(text, p):
    words = text[:-1].split()
    result = []
    
    for word in words:
        if word.isdigit():
            result.append(p.number_to_words(word)) 
        else:
            result.append(word)
    
    return ' '.join(result) + text[-1]

def calculate_wer(item):
    p = inflect.engine()
    text1 = convert_numbers_to_words(item['answer'].strip().lower(), p)
    text2 = convert_numbers_to_words(item['reference'].strip().lower(), p)
    return jiwer.wer(text1, text2)

def eval_wer(args):
    output_file = os.path.join(args.output_dir, 'result_wer.jsonl')
    sum_wer = 0
    with open(args.question, 'r') as f:
        length = sum([1 for _ in f])
    with open(args.answer, 'r') as pt, open(args.reference, 'r') as gt, jsonlines.open(output_file, mode='w') as ot:
        for i, (answer, reference) in tqdm(enumerate(zip(jsonlines.Reader(pt), jsonlines.Reader(gt))), total=length):
            item = {"answer": answer[str(i).zfill(4)], "reference": reference[str(i).zfill(4)]}
            wer = calculate_wer(item)
            item["WER"] = wer
            ot.write(item)
            sum_wer += item["WER"]
        ot.write({"final_WER": sum_wer / length})
    # save results
    logging.info(f"saving result to {output_file}")
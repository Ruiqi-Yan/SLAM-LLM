import jiwer
from tqdm import tqdm
from argparse import ArgumentParser
import logging
import os
import jsonlines
import inflect
import string
import multiprocessing as mp
import numpy as np
import torch.nn.functional as F

def run_asr_wer(lang, item):
    if lang not in ["zh", "en"]:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation

    from jiwer import compute_measures

    truth = item["text"]
    hypo = item["asr"]

    for x in punctuation_all:
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()

    measures = compute_measures(truth, hypo)
    wer = measures["wer"]

    # ref_list = truth.split(" ")
    # subs = measures["substitutions"] / len(ref_list)
    # dele = measures["deletions"] / len(ref_list)
    # inse = measures["insertions"] / len(ref_list)

    return wer

def eval_wav(args):
    wers = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "result_wer.jsonl")

    logging.info("<------start wer eval------>")
    with open(args.answer, 'r') as f:
        length = sum([1 for _ in f])

    with open(args.answer_text, 'r') as gt, open(args.answer, 'r') as pt, jsonlines.open(output_file, mode='w') as ot:
        for i, (text, asr) in tqdm(enumerate(zip(jsonlines.Reader(gt), jsonlines.Reader(pt))), total=length):
            item = {"text": text[str(i).zfill(4)], "asr": asr[str(i).zfill(4)]}
            wer = run_asr_wer(args.language, item)
            item["WER"] = wer
            wers.append(wer)
            ot.write(item)
        wer = np.mean(wers)
        ot.write({"final_WER": wer})

    logging.info(f"Total {len(wers)} samples")
    logging.info(f"WER: {round(wer * 100, 3)}%")
    logging.info(f"saving result to {output_file}")

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

def eval_repeat(args):
    output_file = os.path.join(args.output_dir, 'result_repeat_wer.jsonl')
    sum_wer = 0
    ok_num = 0
    with open(args.question, 'r') as f:
        length = sum([1 for _ in f])
    with open(args.answer, 'r') as pt, open(args.reference, 'r') as gt, jsonlines.open(output_file, mode='w') as ot:
        for i, (answer, reference) in tqdm(enumerate(zip(jsonlines.Reader(pt), jsonlines.Reader(gt))), total=length):
            item = {"answer": answer[str(i).zfill(4)], "reference": reference[str(i).zfill(4)]}
            wer = calculate_wer(item)
            item["WER"] = wer
            if wer <= 0.5:
                ok_num += 1
                sum_wer += item["WER"]
            ot.write(item)
        ot.write({"ok_rate": ok_num / length, "final_WER_for_ok_case": sum_wer / ok_num})
    # save results
    logging.info(f"saving result to {output_file}")
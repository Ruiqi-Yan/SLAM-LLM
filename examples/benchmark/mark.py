from tqdm import tqdm
import multiprocessing
from openai import OpenAI
from argparse import ArgumentParser
import logging
import os
import jsonlines
import gpt_mark
import eval_metrics

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=["qa", "open", "semi-open", "contrast", "wer"])
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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # evaluate data
    logging.info("<========start to evaluate data========>")
    if args.mode == 'wer':
        eval_metrics.eval_wer(args)
    else: gpt_mark.eval(args)

if __name__ == '__main__':
    main()
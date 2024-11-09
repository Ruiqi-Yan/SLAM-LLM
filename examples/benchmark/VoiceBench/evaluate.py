from argparse import ArgumentParser
import json
from src.evaluator import evaluator_mapping
import logging

def main():
    parser = ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--evaluator', type=str, required=True, choices=list(evaluator_mapping.keys()))
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
		level=logging.INFO, 
		format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)

    data = []
    with open(args.src_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)
    evaluator = evaluator_mapping[args.evaluator]()
    final_score = {'final_score': evaluator.evaluate(data)}

    with open(args.src_file, 'a') as f:
        f.write(json.dumps(final_score) + "\n")

    logging.info(f"final score on dataset {args.dataset} is {final_score['final_score']}, saved to {args.src_file}")

if __name__ == "__main__":
    main()

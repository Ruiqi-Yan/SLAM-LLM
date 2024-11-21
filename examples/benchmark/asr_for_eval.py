import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from argparse import ArgumentParser
import os
from tqdm import tqdm
import logging

def set_whisper(model_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_dir)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
		level=logging.INFO, 
		format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)

    pipe = set_whisper(args.model_dir)

    # ASR
    logging.info(f"<========ASR starts========>")
    with open(os.path.join(args.output_dir, "asr_text"), 'w') as f:
        for i in tqdm(range(args.number)):
            audio_file = os.path.join(args.input_dir, (str(i).zfill(4) + ".wav"))
            if os.path.exists(audio_file):
                result = pipe([audio_file], batch_size=1)
                f.write(str(i).zfill(4) + '\t' + result[0]['text'] + '\n')
            else: 
                result = " "
                f.write(str(i).zfill(4) + '\t' + result + '\n')
            
if __name__ == '__main__':
    main()
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from argparse import ArgumentParser
import os

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
    args = parser.parse_args()

    pipe =set_whisper(args.model_dir)
    with open(os.path.join(args.output_dir, "asr_text"), 'w') as f:
        for i in range(2 * len(os.listdir(args.input_dir))):
            audio_file=os.path.join(args.input_dir, (str(i) + ".wav"))
            if os.path.exists(audio_file):
                result = pipe([audio_file])
                f.write(str(i) + '\t' + result + '\n')
            
if __name__ == '__main__':
    main()
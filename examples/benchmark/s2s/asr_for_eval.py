import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from argparse import ArgumentParser

def set_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

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
    parser.add_argument('--path', required=True)
    parser.add_argument('--answer', required=True)
    args = parser.parse_args()

    pipe =set_whisper()
    result = pipe(["audio_1.mp3", "audio_2.mp3"], batch_size=2)

    print(result["text"])

if __name__ == '__main__':
    main()
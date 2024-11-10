import os
import pandas as pd
from pydub import AudioSegment
import io
import jsonlines

# read parquet file
def read_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    return df

# save as wav files
def save_audio(audio_data, output_path):
    audio = AudioSegment.from_file(io.BytesIO(audio_data['bytes']))
    audio.export(output_path, format="wav")

def process_audio_data(parquet_file, output_dir, jsonl_file):
    df = read_parquet(parquet_file)

    os.makedirs(output_dir, exist_ok=True)

    with jsonlines.open(jsonl_file, mode='w') as writer:
        for idx, row in df.iterrows():
            audio_data = row['audio']
            prompt_text = row['prompt']
            
            output_path = os.path.join(output_dir, f"audio_{str(idx).zfill(4)}.wav")
            
            save_audio(audio_data, output_path)
            
            # write as JSONL file
            if "reference" in row:
                reference_text = row['reference']
                writer.write({
                "source_wav": output_path,
                "source_text": prompt_text,
                "target_text": reference_text
            })
            else:
                writer.write({
                "source_wav": output_path,
                "source_text": prompt_text
            })
            print(f"Processed: {output_path}")

def main():
    parquet_file = "/data/ruiqi.yan/data/voicebench/sd-qa/usa-00000-of-00001.parquet"
    output_dir = "/data/ruiqi.yan/data/voicebench_raw/sd-qa/audio"
    jsonl_file = "/data/ruiqi.yan/data/voicebench_raw/sd-qa/test.jsonl"

    process_audio_data(parquet_file, output_dir, jsonl_file)

if __name__ == '__main__':
    main()
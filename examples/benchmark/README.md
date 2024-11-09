# s2s-Benchmark

## Environment Setup
Set up the environment using the following command after setting up the environment for SLAM-LLM:
```bash
# there may be conflicts, but runs well on my machine 
pip install -r requirements.txt
# or
pip install -r requirements.txt --no-dependencies   
```
or you can set up another environment, read [voicebench](VoiceBench/README.md) for more detail. This way, you need to change your environment between inference and marking.


## Evaluation

### non-asr mode
In non-asr mode, we directly evaluate the output text of LLM.

Run the following command:
```bash
# choose ${val_data_name} in (alpacaeval，commoneval，sd-qa)
bash ./s2s/scripts/eval/eval.sh
```
or run inference and marking separately
```bash
# choose ${val_data_name} in (alpacaeval，commoneval，sd-qa)
bash ./s2s/scripts/eval/inference_for_eval_group2.sh
conda activate voicebench
bash ./s2s/scripts/eval/mark_only.sh
```

### asr mode
In asr mode, we use [whisper-large-v3](https://github.com/openai/whisper) for asr and evaluate the transcription of the output speech.

Run the following command:
```bash
# choose ${val_data_name} in (alpacaeval，commoneval，sd-qa)
bash ./s2s/scripts/eval/eval_with_asr.sh
```
or run inference and marking separately
```bash
# choose ${val_data_name} in (alpacaeval，commoneval，sd-qa)
bash ./s2s/scripts/eval/inference_for_eval_group2.sh
conda activate voicebench
bash ./s2s/scripts/eval/asr_for_eval.sh
```
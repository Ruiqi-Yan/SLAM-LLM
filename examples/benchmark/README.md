# s2s-Benchmark

## Environment Setup
Set up the environment using the following command after setting up the environment for SLAM-LLM:
```bash
# there may be conflicts, but runs well on my machine 
pip install -r requirements.txt   
```
or you can set up another environment, read [voicebench](VoiceBench/README.md) for more detail. This way, you need to change your environment between inference and marking.


## Evaluation
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

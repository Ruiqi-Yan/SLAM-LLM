#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/LLaMA-Omni-test

# jsonl dataset
manifest_format=jsonl
val_data_name="alpacaeval"     # alpacaeval，commoneval，sd-qa
val_data_path=/data/ruiqi.yan/data/voicebench_raw/${val_data_name}/test.jsonl

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/LLaMA-Omni-test/${val_data_name}

# huggingface dataset
# manifest_format=datasets
# val_data_path="/data/ruiqi.yan/data/voicebench"
# val_data_name="alpacaeval"     # alpacaeval，commoneval，sd-qa
# load_from_cache_file=true
# dataset_sample_seed=888


cd $ckpt_dir
source /home/visitor/miniconda3/etc/profile.d/conda.sh
conda activate yrq-llama-omni          # put your environment name here
# -m debugpy --listen 5678 --wait-for-client
# python $code_dir/LLaMA-Omni-test/inference_for_eval.py \
#         --dataset $val_data_path \
#         --results-path $decode_log \
#         --s2s \
#         --dur-prediction \
#         --model-path $ckpt_dir/models/Llama-3.1-8B-Omni \
#         --vocoder $ckpt_dir/vocoder/g_00500000 \
#         --vocoder-cfg $ckpt_dir/vocoder/config.json


source /home/visitor/miniconda3/etc/profile.d/conda.sh
conda activate yrq-omni          # put your environment name here
output_dir=$decode_log/eval/${val_data_name}
# data_number=199         # 199, 200, 553 for alpacaeval，commoneval，sd-qa

# python $code_dir/s2s/asr_for_eval.py \
#         --input_dir $decode_log/audio \
#         --model_dir "/data/yanruiqi/model/whisper-large-v3" \
#         --output_dir $decode_log \
#         --number $data_number

if [ "$val_data_name" = "sd-qa" ]; then
    evaluator="qa"
    python $code_dir/VoiceBench/api_judge.py \
        --question $decode_log/question_text \
        --answer $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --reference $decode_log/gt_text
fi

if [ "$val_data_name" != "sd-qa" ]; then
    evaluator="open"
    python $code_dir/VoiceBench/api_judge.py \
        --question $decode_log/question_text \
        --answer $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name
fi

python $code_dir/VoiceBench/evaluate.py \
        --src_file $output_dir/result.jsonl \
        --evaluator $evaluator \
        --dataset $val_data_name
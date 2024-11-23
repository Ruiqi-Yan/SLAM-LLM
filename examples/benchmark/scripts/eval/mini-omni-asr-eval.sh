#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

# jsonl dataset
manifest_format=jsonl
# alpacaeval_test, commoneval_test, gaokao_test, gsm8k_test, mlc_test, repeat_test, storal_test, summary_test, truthful_test, wildchat_test
val_data_name="gaokao_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=303         # 199, 200, 303 for alpacaeval，commoneval，gaokao
error_list="28,237"     # "13,52", "", "84,390,418" for alpacaeval，commoneval，sd-qa

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# huggingface dataset
# manifest_format=datasets
# val_data_path="/data/ruiqi.yan/data/voicebench"
# val_data_name="alpacaeval"     # alpacaeval，commoneval，sd-qa
# load_from_cache_file=true
# dataset_sample_seed=888


# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir \
        --error_list $error_list


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
mode="qa"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --reference $decode_log/gt_text
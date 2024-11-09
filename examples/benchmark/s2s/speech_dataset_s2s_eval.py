import json
import copy

import numpy as np

import torch
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d
from utils.snac_utils import layershift, get_snac_answer_token, simple_shift
from utils.codec_utils import get_single_layer_answer_token, get_group_answer_token
import librosa

from datasets import load_dataset, load_from_disk
import datasets

ds = load_dataset("hlt-lab/voicebench", "alpacaeval", split="test")
ds.to_json("/data/yanruiqi/SLAM-LLM/examples/benchmark/data/alpacaeval.json")

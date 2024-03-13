# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from slam_llm.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from slam_llm.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from slam_llm.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
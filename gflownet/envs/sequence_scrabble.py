"""
Environment to test sequence scrabble
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.scrabble import Scrabble

from gflownet.utils.common import copy


class SequenceScrabble(Sequence):
    """
    Environment to test multiple scrabble
    """

    def __init__(
        self,
        **kwargs,
    ):

        # Initialize list of subenvs:
        subenvs = [
            Scrabble(letters=["D","G","O"]),
            Scrabble(letters=["A", "C", "T"]),
            Scrabble(letters=["F", "H", "I", "S"])
        ]

        # Initialize base Stack environment
        super().__init__(envs_unique=tuple(subenvs), subenvs=tuple(subenvs), **kwargs)

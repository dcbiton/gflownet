"""
Environment to test sequence scrabble
"""

from typing import Iterable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.base import GFlowNetEnv
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

    def _get_unique_environments(self, subenvs):
        envs_unique = []
        envs_unique_keys = []
        unique_indices = []
        for env in subenvs:
            env_key = (type(env), tuple(env.action_space), tuple(env.letters))
            if env_key not in envs_unique_keys:
                envs_unique_keys.append(env_key)
                envs_unique.append(env)
            unique_indices.append(envs_unique_keys.index(env_key))
        return envs_unique, tuple(envs_unique_keys), unique_indices
    
    def _compute_unique_indices_of_subenvs(
            self, subenvs: Iterable[GFlowNetEnv]
        ) -> List[int]:
            """
            Returns the list of unique-environment indices corresponding to each
            sub-environment in ``subenvs``.
            """
            indices_unique = []
            for env in subenvs:
                try:
                    indices_unique.append(
                        self.envs_unique_keys.index((type(env), tuple(env.action_space), tuple(env.letters)))
                    )
                except ValueError:
                    raise ValueError(
                        "The list of subenvs contains a sub-environment that could not "
                        "be matched to one of the existing unique environments"
                    )
            return indices_unique
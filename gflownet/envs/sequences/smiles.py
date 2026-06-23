"""Class to represent SMILES molecules."""

from typing import List, Optional, Union

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.sequences.base import SequenceBase

# from gflownet.envs.sequences.smiles_organic_vocab import vocab as SMILES_VOCAB
from gflownet.envs.sequences.smiles_organic_vocab_genetic_gfn import (
    vocab as SMILES_VOCAB,
)
from gflownet.utils.common import tlong


class Smiles(SequenceBase):
    """
    A SMILES sequence environment whose tokens are random SMILES tokens.

    Smiles may not be a valid molecule
    """

    def __init__(
        self,
        smiles_vocab: List[str] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        smiles_vocab : List[str] | None
            The list of SMILES tokens to use as the vocabulary.
            If None (default), the small vocabulary defined in
            SMILES_VOCAB is used.
        """
        if smiles_vocab is None:
            smiles_vocab = SMILES_VOCAB
        self.smiles_vocab = smiles_vocab
        super().__init__(tokens=self.smiles_vocab, **kwargs)

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]],  # noqa: F821
            TensorType["batch", "max_length"],  # noqa: F821
        ],
    ) -> List[str]:
        """
        Prepare a batch of states for a SMILES-string proxy.

        The proxy representation is the compact SMILES string obtained by
        concatenating all non-padding tokens in the sequence.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A list containing one SMILES string per state.
        """
        states = tlong(states, device=self.device).tolist()
        return [
            "".join(self.idx2token[idx] for idx in self._unpad(state))
            for state in states
        ]

    def state2readable(
        self, state: Optional[TensorType["max_length"]] = None  # noqa: F821
    ) -> str:
        """
        Convert a state into a human-readable string.

        Example, with max_length = 5:
          - state: [1, 2, 1, 1, 0]
          - readable: "0 1 0 0"

        The output string contains the token corresponding to each index in the state,
        separated by spaces.

        Parameters
        ----------
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated tokens.
        """
        state = self._get_state(state)
        state = self._unpad(state.tolist() if torch.is_tensor(state) else state)
        return "".join([str(self.idx2token[idx]) for idx in state])[:-1]

"""
An environment to generate a Branch SELFIES token

From the source state, we can choose among 6 actions until the end-of-sequence action
after choosing a "Branch" action, only then will be the end-of-sequence action be valid

actions will be:
(1) Branch
(2) =Branch
(3) #Branch
(4) 1
(5) 2
(6) 3
After 1,2,or 3 then the EOS action will be valid

"""

import re
from typing import Iterable, List, Optional, Tuple

import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong


class Branch(GFlowNetEnv):
    """
    Branch environment for generating a branch SELFIES token
    States are represented by a list of 2 elements, corresponding to 2 descriptors:

    - Bond (1) [optional]
    - Next token (2) [optional]
        |-- Next token is needed by the branch to look at the next token included in the branch
    When (2)nd element is not null, EOS action will be valid

    Actions are represented as tuples with two elements:

    - Descriptor type: one of the three descriptors of states
    - Value: the specific value of the descriptor

    Attributes
    ----------
    source_readable : str
        The string to be used to represent the source state as a human-readable string.
    """

    def __init__(
        self,
        source_readable: str = "<source>",
        **kwargs,
    ):
        """
        Initialize a Branch environment.

        Parameters
        ----------
        source_readable : str
            The string to be used to represent the source state as a human-readable
            string. By default: <source>
        """
        self.source_readable = source_readable
        # Source state: [bond, next_token]
        self.source = [0, 0]
        # End-of-sequence action
        # (action_type, action_value)
        self.eos = (-1, -1)
        # readable format for the branch environment, hard-code set-up
        self.branch_readable = self.state2readable(state=self.source)
        # action map
        self.action_map = {"bond": ["=", "#"], "next_token": ["1", "2", "3"]}
        # count the number of actions per action_type
        # will be used later in masks
        self.num_actions = {
            action_type: len(self.action_map[action_type])
            for action_type in self.action_map
        }
        # cummulative count will be used for masking later
        self.cummulative_count_actions = {
            action_type: sum(
                [
                    self.num_actions[action_type_iter]
                    for i, action_type_iter in enumerate(self.num_actions)
                    if i < j
                ]
            )
            for j, action_type in enumerate(self.num_actions)
        }
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Return all the possible actions for the branch including EOS.

        Actions are represented by action = (action_type, action_value)
        action types are:
        (1) bond (=#)            [optional]
        (2) next token           [required]

        The end of sequence action is (-1, -1).

        for each action type:
        (1) bond: = #
        (2) next_token: 1 2 3

        Returns
        -------
        list
            A list of tuples representing the actions.
        """
        action_space = []
        for i, action_type in enumerate(self.action_map):
            action_space += [
                (
                    i + 1,
                    action_value + 1,
                )
                for action_value in range(len(self.action_map[action_type]))
            ]
        action_space += [self.eos]
        return action_space

    def _is_bond_set(self, state: List[int]) -> bool:
        """
        Checks if the bond is set
        By looking at state[0]
        If state[0] == 0 then there is no bond yet
        else if state[0] != 0 then a bond was chosen
        """
        if state[0] != 0:
            return True
        else:
            return False

    def _is_next_token_set(self, state: List[int]) -> bool:
        """
        Checks if the next_token indicator is set
        By looking at state[1]
        If state[1] == 0 then there is no next_token yet
        else if state[1] != 0 then next_token was chosen
        """
        if state[1] != 0:
            return True
        else:
            return False

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns which actions are invalid (True) and which are not invalid (False).

        Parameters
        ----------
        state : list
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        num_bonds = self.num_actions["bond"]
        num_next_token = self.num_actions["next_token"]

        # for the mask, all are valid unless it is already set
        if self._is_bond_set(state):
            masked_bonds = [True] * num_bonds
        else:
            masked_bonds = [False] * num_bonds

        # Determine which elements are valid
        if self._is_next_token_set(state):
            masked_next_token = [True] * num_next_token
            terminal_mask = [False]
        else:
            masked_next_token = [False] * num_next_token
            terminal_mask = [True]

        # after checking everything,
        # combine lists to create a mask for the whole action space
        mask = masked_bonds + masked_next_token + terminal_mask
        return mask

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determine all parents and actions that lead to state.

        There are many states:
            - Source state: no parents

        Parameters
        ----------
        state : list
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format.

        actions : list
            List of actions that lead to state for each parent in parents.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []

        # container for the parents and action
        parents = []
        p_actions = []

        # all the non-zero states are the number of actions
        for i in range(len(state)):
            if state[i] != 0:
                parent = state.copy()
                parent[i] = 0
                parents.append(parent)
                p_actions.append((i + 1, state[i]))
        return parents, p_actions

    def step(
        self, action: Tuple[int, int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Execute step given an action.

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a tuple with a single element indicating
            the index of the option to be set.
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The state after executing the action.
        action : tuple
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action=action,
            backward=False,
            skip_mask_check=skip_mask_check or self.skip_mask_check,
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1

        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid

        # Update state: simply set the index indicated by the action
        self.state[action[0] - 1] = action[1]
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Return the maximum trajectory length of the environment.

        The maximum trajectory length is 3 including EOS but length could be 2 to 3.
        """
        # (1) Branch action[1,2,3] required
        # (2) action[4,5,6] optional
        # (3) EOS
        return 3

    def states2policy(self, states: List) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Parameters
        ----------
        states : list
            A batch of states in environment format

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)
        states_policy = tlong(states, device=self.device)
        states_policy = F.one_hot(
            states_policy, num_classes=1 + max(self.num_actions.values())
        )
        return tfloat(
            states_policy.reshape(n_states, -1),
            float_type=self.float,
            device=self.device,
        )

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        # get the branch
        m = re.search(r"\[(.?)Branch(.?)\]", readable)
        if m is None:
            return [0, 0]
        bond, next_token = m.groups()
        return [
            0 if bond == "" else 1 + self.action_map["bond"].index(bond),
            (
                0
                if next_token == ""
                else 1 + self.action_map["next_token"].index(next_token)
            ),
        ]

    def state2readable(self, state: Optional[List] = None, alphabet={}):
        """
        Converts a state into a human-readable string representing a state.

        The readable representation is taken from ``self.options``, except if the state
        is the source state in which case ``self.source_readable`` is returned.
        """
        state = self._get_state(state)
        # if the state is the source, return source_readable
        if self.is_source(state):
            return self.source_readable
        bond_branch = "" if state[0] == 0 else self.action_map["bond"][state[0] - 1]
        next_token_branch = (
            "" if state[1] == 0 else self.action_map["next_token"][state[1] - 1]
        )
        readable_state = "[{}Branch{}]".format(bond_branch, next_token_branch)
        print("STATE:", state, "READABLE:", readable_state)
        return readable_state

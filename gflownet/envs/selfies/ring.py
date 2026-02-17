"""
An environment to generate a Ring SELFIES token

From the source state, we can choose among 6 actions until the end-of-sequence action
after choosing a "Ring" action, only then will be the end-of-sequence action be valid

actions will be:
(1) Ring
(2) =Ring
(3) #Ring
(4) -/Ring
(5) -\\Ring
(6) /-Ring
(7) \\-Ring
(8) //Ring
(9) /\\Ring
(10) //Ring
(11) /\\Ring
(12) 1
(13) 2
(14) 3
After 1-11(inc) then the EOS action will be valid

"""

from typing import Iterable, List, Optional, Tuple

import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong


class Ring(GFlowNetEnv):
    """
    Ring environment for generating a branch SELFIES token

    States are represented by a single-element list, indicating the index of the
    selected option (starting from 1), or 0 if no option has been selected yet (source
    state).

    The actions of the environment are single-element tuples indicating the index of
    the option to be selected, or -1 for the EOS.

    Attributes
    ----------
    options : iterable
        The description of the options as an iterable of strings. These strings are
        used as readable representation. By default, the string <source> is reserved
        for the source state.
    n_options : int
        The number of options.
    options_available : tuple
        A tuple of option indices (starting from 1) corresponding to the subset of
        options that is available for the environment instance.
    source_readable : str
        The string to be used to represent the source state as a human-readable string.
    """

    def __init__(
        self,
        source_readable: str = "<source>",
        **kwargs,
    ):
        """
        Initializes a Ring environment.

        Parameters
        ----------
        source_readable : str
            The string to be used to represent the source state as a human-readable
            string. By default: <source>
        """
        self.source_readable = source_readable
        # Source state: [0]
        self.source = []
        # End-of-sequence action
        self.eos = (-1,)
        # readable format for the branch environment, hard-code set-up
        self.ring_readable = [
            "Ring",
            "=Ring",
            "#Ring",
            "-/Ring",
            "-\\Ring",
            "/-Ring",
            "\\-Ring",
            "//Ring",
            "/\\Ring",
            "//Ring",
            "/\\Ring",
            "1",
            "2",
            "3",
        ]

        # keep track of the state of the forward mask depending on the state
        self.forward_mask = [False] * len(self.ring_readable) + [False]
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        returns all the possible actions for the branch including EOS.

        Actions are represented by one element, starting from 1.
        The actual element to action mapping is:
        (1) Ring
        (2) =Ring
        (3) #Ring
        (4) -/Ring
        (5) -\\Ring
        (6) /-Ring
        (7) \\-Ring
        (8) //Ring
        (9) /\\Ring
        (10) //Ring
        (11) /\\Ring
        (12) 1
        (13) 2
        (14) 3
        The end of sequence action is (-1,).

        Returns
        -------
        list
            A list of tuples representing the actions.
        """
        action_space = [(el,) for el in range(1, 15)] + [self.eos]
        return action_space

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

        # If done is True, return all invalid
        if done:
            return [True] * self.action_space_dim

        # If the state is the source state, all actions of available options are valid
        # except EOS
        print("STATE:", state)
        if self.is_source(state):
            return [False] * 14 + [True]

        elif len(state) == 2:
            return [True] * 14 + [False]
        elif state[0] <= 11:
            # [1], [2], [3]
            # if state contains the "Branch" word, then EOS can be valid
            return [True] * 11 + [False] * 4

        elif state[0] > 11:
            # [4], [5], [6]
            # if there is already a number, only the first three actions is valid
            return [False] * 11 + [False] * 4

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

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
        # THIS IS NOT DONE YET
        if done:
            return [state], [self.eos]
        elif self.is_source(state):
            return [], []
        else:
            return [state[:-1]], [(state[-1],)]

    def step(
        self, action: Tuple[int, int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

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
        self.state.append(action[0])
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory length is 3 including EOS but length could be 2 to 3.
        """
        # (1) Ring action 1-11 required
        # (2) action 12-14 optional
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
        states_policy = F.one_hot(states_policy, num_classes=7)
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
        if readable == self.source_readable:
            return self.source
        return [self.ring_readable.index(readable) + 1]

    def state2readable(self, state: Optional[List] = None, alphabet={}):
        """
        Converts a state into a human-readable string representing a state.

        The readable representation is taken from ``self.options``, except if the state
        is the source state in which case ``self.source_readable`` is returned.
        """
        state = self._get_state(state)
        state.sort()
        # if the state is the source, return source_readable
        if self.is_source(state):
            return self.source_readable
        return (
            "["
            + "".join([str(self.ring_readable[idx - 1]) + "" for idx in state])
            + "]"
        )

    def get_all_terminating_states(self) -> List[List[int]]:
        """
        Returns a list with all the terminating states in environment format.

        Returns
        -------
        list
            The list of all terminating states.
        """
        return [[idx] for idx in range(1, 15)]

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Parameters
        ----------
        n_states : int
            The number of states to sample.
        seed : int
            Random seed.
        """
        random.seed(seed)
        return random.choices(self.get_all_terminating_states(), k=n_states)

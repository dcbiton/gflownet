"""
An environment to generate an Atom SELFIES token.

States are represented as a list of 6 elements, which correspond to the 6 descriptors
that can appear in an atom token:

- Bond (1) [optional]
- Isotope (2) number [optional]
- Element (3) [required]
- Chiral tag (4) [optional]
- Hydrogen count (5) [optional]
- Charge (6) [optional]
When (3)rd element is not null, EOS action will be valid

Actions are represented as tuples with two elements:

- Descriptor type: one of the six descriptors of states
- Value: the specific value of the descriptor

Explicitly the state and action are:
state = [bond, isotope, element, chiral_tag, H_count, charge]
action = (action_type, action_value)

Bond descriptors can have 2 values:

- Double bond, = (1)
- Triple bond, # (2)
- Stereochemistry: / (3)
- Stereochemistry: / (4)

If no bond descriptor is explicitly included in the token, single bond is assumed.

Element descriptors can have as many values as elements in the periodic table. The
value corresponds to the atomic number of the element. Valid elements can be restricted
at initialization of the environment.

From the source state, we can choose among the actions until the end-of-sequence action
after choosing an "Atom" action, only then will be the end-of-sequence action be valid

actions will be: choose a:
(1) bond (=#/\\)            [optional]
(2) isotope number ()       [optional]
(3) element
(4) chiral tag  (@, @@)    [optional]
(5) H count (H1, H3, etc)   [optional]
(6) charge ([+-][1-9])      [optional] {largest is Xe+38}

[CURRENT IMPLEMENTATION] Valid elements:
all charged elements are valid, only bond constraints are incorporated
If the charged_element doesn't have a bond  constraint,
then the bond constraint is 8
This is implemented only for terminating states (TODO: should be implemented to masking as well)

"""

import re
from itertools import product
from typing import Iterable, List, Optional, Tuple, Union

import torch.nn.functional as F
from selfies.bond_constraints import _current_constraints
from selfies.constants import (
    ELEMENTS,  # we don't use this constant since it is not ordered well
)
from selfies.constants import (
    INDEX_ALPHABET,
    INDEX_CODE,
    ORGANIC_SUBSET,
)
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, tlong

# map of element name to element number based on periodic table
element_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
}

organic_subset_map = {
    e: element_map[e] for e in element_map.keys() if e in list(ORGANIC_SUBSET) + ["H"]
}

bond_constraints = _current_constraints


class Atom(GFlowNetEnv):
    """
    Atom environment for generating an Atom SELFIES token

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
    elements : list or int
        Elements that will be used for construction of crystal. Either list, in
        which case every value should indicate the atomic number of an element, or
        int, in which case n consecutive atomic numbers will be used. Note that we
        assume this will correspond to real atomic numbers, i.e. start from 1, not
        0.

    optional paratemeters in the state:
    allow_isotopes: bool
        Allow the state to consider varying isotopes
    allow_chiral_tag: bool
        Allow the state to consider chiral tag
    allow_h_count: bool
        Allow the state to consider different H counts

    limits in the state space
    isotope_limit: int
        Maximum isotope number to be considered by all elements
        from 0 to isotope_limit (inclusive)
    h_count_limit: int
        Maximum Hydrogen count to be considered by all elements
        from 0 to h_count_limit (inclusive)
    charge_limit: int
        Maximum charge possible for all elements
        from -charge_limit to +charge_limit

    """

    def __init__(
        self,
        elements: Union[List, str, int] = "organic",
        allow_chiral_tag: bool = False,
        allow_stereoisomer: bool = False,
        isotope_limit: int = 0,
        h_count_limit: int = 0,
        charge_limit: int = 4,
        source_readable: str = "<source>",
        **kwargs,
    ):
        """
        Initializes an Atom environment.

        Parameters
        ----------
        source_readable : str
            The string to be used to represent the source state as a human-readable
            string. By default: <source>
        """
        if isinstance(elements, int):
            # use element_map to keep the order of the elements according to the atomic number
            self.elements = {
                element: i for element, i in element_map.items() if i < elements
            }
        elif isinstance(elements, List):
            # use element_map to keep the order of the elements according to the atomic number
            self.elements = {
                element: i for element, i in element_map.items() if element in elements
            }
        elif isinstance(elements, str):
            if elements == "organic":
                self.elements = organic_subset_map
            elif elements == "all":
                self.elements = element_map
            else:
                raise ValueError(
                    f'Provided elements must be list, int, or str("organic", "all")'
                )

        self.source_readable = source_readable
        # Source state: [bond, isotope, element, chiral_tag, H_count, charge]
        self.source = [0, 0, 0, 0, 0, 0]
        # End-of-sequence action
        self.eos = (
            -1,
            -1,
        )  # (action_type, action_value)
        self.atom_readable = self.state2readable(state=self.source)
        # action map
        self.action_map = {
            "bond": ["=", "#", "/", "\\"] if allow_stereoisomer else ["=", "#"],
            "isotope_number": [str(i) for i in range(1, isotope_limit + 1)],
            "element": list(self.elements.keys()),
            "chiral_tag": ["@", "@@"] if allow_chiral_tag else [],
            "h_count": (
                ["H" + str(i) for i in range(1, h_count_limit + 1)]
                if h_count_limit > 0
                else []
            ),
            "charge": [str(i) for i in range(-charge_limit, 0)]
            + ["+" + str(i) for i in range(1, charge_limit + 1)],
        }
        # map the bond to the number of bonds
        self.bond_map = {"=": 2, "#": 3, "/": 2, "\\": 3}
        # element to index map that will be used to for creating a mask
        self.element_to_action_index = {
            element: i for i, element in enumerate(self.action_map["element"])
        }
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
        # used for getting bond constraints of each element (charged or uncharged)
        self.element_to_bond_constraint = {
            self.element_to_action_index[element]: bond_constraints[element]
            for element in self.action_map["element"]
        }
        self.charged_element_bond_constraints = {
            self.element_to_action_index[element]: {
                i: bond_constraints[element + charge]
                for i, charge in enumerate(self.action_map["charge"])
                if element + charge in bond_constraints.keys()
            }
            for element in self.action_map["element"]
        }
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        returns all the possible actions for the atom including EOS.

        Actions are represented by action = (action_type, action_value) 
        action types are: 
        (1) bond (=#/\\)            [optional]
        (2) isotope number ()       [optional]
        (3) element
        (4) chiral tag (, @, @@)    [optional]
        (5) H count (H1, H3, etc)   [optional]
        (6) charge ([+-][1-9])      [optional] {largest is Xe+38}

        The end of sequence action is (-1, -1).

        for each action type: 
        (1) bond: = # / \\ 
        (2) isotope_number: minimum 0 maximum: 250
        (3) element: based on self.element 
        (4) chiral_tag: (, @, @@)
        (5) h_count: minimum=0 maximum 4
        (6) charge: minimum=-38 maximum=38

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

        # for the mask, we limit the action space based on both
        # the action_type and action_value of the action_map

        # If done is True, all actions are invalid
        if done:
            mask = [True] * self.action_space_dim
            return mask

        if self.is_source(state):
            # If the state is the source state, all actions of available options are valid
            # except EOS
            mask = [False] * (self.action_space_dim - 1) + [True]
            return mask

        # if parts of the state has a non-source/non-zero value,
        # only EOS is valid
        elif 0 not in state:
            # all states are invalid except the EOS
            mask = [True] * (self.action_space_dim - 1) + [False]
            return mask

        mask = [False] * (self.action_space_dim - 1) + [True]
        # next if state[2] (which refers to the element) has a non-source/non-zero value,
        # EOS is valid
        if state[2] != 0:
            mask[-1] = False
            # check the atom and apply bond constraints

        for action_ind, action_type in enumerate(self.action_map):
            if state[action_ind] != 0:
                # if the state is already filled with an action_type,
                # actions with that action_type is masked
                mask_start = self.cummulative_count_actions[action_type]
                mask_length = self.num_actions[action_type]
                mask[mask_start : mask_start + mask_length] = [True] * mask_length

        # to incorporate the bond constraints, count the bond possible for each element
        # if there is a chemical element but no bonds yet
        # BOND:No ELEMENT:Yes CHARGE:Yes/No
        if (state[2] != 0) and (state[0] == 0):

            if state[5] != 0:
                # if there is a charge, concatenate the element and the charge
                element_bond_constraint = bond_constraints.get(
                    self.action_map["element"][state[2] - 1]
                    + self.action_map["charge"][state[5] - 1]
                )
            else:
                element_bond_constraint = bond_constraints.get(
                    self.action_map["element"][state[2] - 1]
                )
                if element_bond_constraint < 2:  # only organic elements are considered
                    # if the bond_constrain = 1 then the element cannot be charged (in a molecule)
                    # mask the charges
                    mask_start = self.cummulative_count_actions["charge"]
                    mask_length = self.num_actions["charge"]
                    mask[mask_start : mask_start + mask_length] = [True] * mask_length
            if element_bond_constraint != None:
                # mask some of the "bond" actions based on the bond constraints
                mask_start = (
                    self.cummulative_count_actions["bond"]
                    + min(3, element_bond_constraint)
                    - 1
                )
                mask_length = (
                    self.num_actions[action_type] - min(3, element_bond_constraint) + 1
                )
                # mask all the bonds based on the bond constraints
                mask[mask_start : mask_start + mask_length] = [True] * mask_length

        # BOND:Yes ELEMENT:Yes CHARGE:No
        if ((state[0] != 0) and (state[2] != 0)) and (state[5] == 0):
            # first, get the chosen bond
            bond_number = self.bond_map[self.action_map["bond"][state[0] - 1]]
            # for all possible charges get the bond contraint of that by concatenating
            # if the bond constraint is less than bond in state[0] then the mask for that action is invalid
            charge_mask = [
                bond_number
                > bond_constraints.get(
                    self.action_map["element"][state[2] - 1]
                    + self.action_map["charge"][charge_i],
                    9,
                )
                for charge_i in range(0, (self.num_actions["charge"]))
            ]
            # apply the mask
            mask_start = self.cummulative_count_actions["charge"]
            mask_length = self.num_actions["charge"]
            mask[mask_start : mask_start + mask_length] = charge_mask

        # add a mask for elements after a bond is chosen
        # if there is a bond but there is no element yet
        # BOND:Yes/No ELEMENT:No CHARGE:No
        if (state[2] == 0) and (state[5] == 0):
            # filter actions for some elements
            # first, get the bond
            bond_number = (
                self.bond_map[self.action_map["bond"][state[0] - 1]]
                if state[0] > 0
                else 0
            )
            # invalid elements are those with bond contraints smaller than the chosen bond number
            invalid_elements = [
                element
                for element, bond in bond_constraints.items()
                if ((bond < bond_number) and (not re.compile(r"[+-?]").search(element)))
            ]
            mask_start = self.cummulative_count_actions["element"]
            # loop through all the elements (Use a loop for now)
            for x in invalid_elements:
                mask[mask_start + self.element_to_action_index[x]] = True

        # if there is no bonds and no elements but there is a change
        # BOND:No/Yes ELEMENT:No CHARGE:Yes
        # state[5] is charge
        # state[0] is bond
        # state[2] is element
        if (state[5] != 0) and (state[2] == 0):
            # the charge in the state
            atom_charge_str = self.action_map["charge"][state[5] - 1]
            atom_charge = abs(int(atom_charge_str))
            # filter out elements
            mask_start = self.cummulative_count_actions["element"]
            if state[0] == 0:
                # filter the elements with bond constraints less than the chosen charge
                invalid_elements = [
                    element
                    for element, bond in bond_constraints.items()
                    if (
                        (bond <= atom_charge)
                        and (not re.compile(r"[+-?]").search(element))
                    )
                ]
                for x in invalid_elements:
                    mask[mask_start + self.element_to_action_index[x]] = True
            else:
                # add the bond number to the bond filter
                # since there is already a charge, loop through all the elements and check the bond constraint
                # chosen bond
                bond_number = self.bond_map[self.action_map["bond"][state[0] - 1]]
                valid_elements = [
                    element.replace(atom_charge_str, "")
                    for element, bond in bond_constraints.items()
                    if (
                        (bond >= bond_number)  # inclusive
                        and (atom_charge_str in element)
                    )
                ]
                # filter-out the elements from the possible actions
                mask_length = self.num_actions["element"]
                mask[mask_start : mask_start + mask_length] = [True] * mask_length
                for x in valid_elements:
                    mask[mask_start + self.element_to_action_index[x]] = False

        return mask

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
        self.state[action[0] - 1] = action[1]
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory length is 7 including EOS
        """
        # (1) bond (=#/\\)            [optional]
        # (2) isotope number ()       [optional]
        # (3) element
        # (4) chiral tag (, @, @@)    [optional]
        # (5) H count (H1, H3, etc)   [optional]
        # (6) charge ([+-][1-9])      [optional] {largest is Xe+38}
        # (7) EOS
        return 7

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
            states_policy,
            num_classes=self.cummulative_count_actions["charge"]
            + self.num_actions["charge"],
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
        if readable == self.source_readable:
            return self.source
        # used the same decoder from selfies
        SELFIES_ATOM_PATTERN = re.compile(
            r"^[\[]"  # opening square bracket [
            r"([=#/\\]?)"  # bond char
            r"(\d*)"  # isotope number (optional, e.g. 123, 26)
            r"([A-Z]?[a-z]?)"  # element symbol
            r"([@]{0,2})"  # chiral_tag (optional, only @ and @@ supported)
            r"((?:[H]\d)?)"  # H count (optional, e.g. H1, H3)
            r"((?:[+-][1-9]+)?)"  # charge (optional, e.g. +1)
            r"[]]$"  # closing square bracket ]
        )
        m = SELFIES_ATOM_PATTERN.match(readable)
        if m is None:
            return [0, 0, 0, 0, 0, 0]
        bond_char, isotope, element, chirality, h_count, charge = m.groups()
        state_str = [bond_char, isotope, element, chirality, h_count, charge]
        return [
            (
                0
                if state_str[i] == ""
                else 1 + self.action_map[action_type].index(state_str[i])
            )
            for i, action_type in enumerate(self.action_map)
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

        readable_state = (
            "["
            + "".join(
                [
                    str(self.action_map[state_name][state[i] - 1])
                    for i, state_name in enumerate(self.action_map)
                    if state[i] != 0
                ]
            )
            + "]"
        )
        return readable_state

    # TODO: This method is work in progress and need to be tested before using them
    def _get_all_terminating_states_wip(self) -> List[List[int]]:
        """
        Returns a list with all the terminating states in environment format.

        Returns
        -------
        list
            The list of all terminating states.
        """
        # get all the elements
        terminating_states = []
        terminating_substates = [
            (
                list(range(1, self.num_actions[k] + 1))
                if k == "element"
                else list(range(0, self.num_actions[k] + 1))
            )
            for k in self.action_map.keys()
        ]
        # get cartertesian product of all the optional states with any element
        # check the bond constraints of the element with a combination with all the charges
        # if there are no bond constraints, then the bond constraint is 8
        for state_tuple in product(*terminating_substates):
            # incorporate the bond constraints
            # if (bool(charged_element_bond_constraints[state_tuple[2]-1]) and (charged_element_bond_constraints[state_tuple[2]-1].get(state_tuple[5]-1)) ) and ((state_tuple[5]!=0) and (state_tuple[0] !=0)):
            if (
                self.charged_element_bond_constraints[state_tuple[2] - 1].get(
                    state_tuple[5] - 1
                )
            ) and ((state_tuple[5] != 0) and (state_tuple[0] != 0)):
                # get the charged  constraint
                bond_string = self.action_map["bond"][state_tuple[0] - 1]
                if (
                    self.bond_map[bond_string]
                    <= self.charged_element_bond_constraints[state_tuple[2] - 1][
                        state_tuple[5] - 1
                    ]
                ):
                    terminating_states.append(list(state_tuple))
            elif bool(self.element_to_bond_constraint[state_tuple[2] - 1]) and (
                state_tuple[0] != 0
            ):
                # get the bond constraint
                bond_string = self.action_map["bond"][state_tuple[0] - 1]
                if (
                    self.bond_map[bond_string]
                    <= self.element_to_bond_constraint[state_tuple[2] - 1]
                ):
                    terminating_states.append(list(state_tuple))
            else:
                # append it to the terminating states
                terminating_states.append(list(state_tuple))

        return terminating_states

    # TODO: This method is work in progress and need to be tested before using them
    def _get_uniform_terminating_states_wip(
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

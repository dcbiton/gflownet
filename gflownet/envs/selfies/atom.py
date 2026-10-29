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
from gflownet.utils.crystals.constants import OXIDATION_STATES  # use this FOR NOW

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
# CHarges considered based on SELFIES
default_charges = {
    "H": [0],
    "F": [0],
    "Cl": [0],
    "Br": [0],
    "I": [0],
    "B": [-1, 0, 1],
    "O": [-1, 0, 1],
    "N": [-1, 0, 1],
    "C": [-1, 0, 1],
    "P": [-1, 0, 1],
    "S": [-1, 0, 1],
}
# bond constraints based on SELFIES
default_bond_constraints = {
    "H": {0: 1},
    "F": {0: 1},
    "Cl": {0: 1},
    "Br": {0: 1},
    "I": {0: 1},
    "B": {0: 3, 1: 2, -1: 4},
    "O": {0: 2, 1: 3, -1: 1},
    "N": {0: 3, 1: 4, -1: 2},
    "C": {0: 4, 1: 3, -1: 3},
    "P": {0: 5, 1: 4, -1: 6},
    "S": {0: 6, 1: 5, -1: 5},
}


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
    source_readable : str
        The string to be used to represent the source state as a human-readable string.
    elements : list or int
        Elements that will be used for construction of token. Either list, in
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
    additional_bond_constraints: dict of dict
        {element(str): {charge(int): max bond possible inclusive (int)}}
        additional bond constraint for the elements, should be in a form of dictionary where the keys are the element and the value is another dictionary where the keys are the charge of the element and the value is the maximum possible bond that the element can accept
        if format of the value is not followed, instantiation will give an error
        additional_bond_constraint should follow the format of default_bond_constraint
        additional_bond_constraint will be appended or will overwrite default_bond_constraint of the given element-charge
        default bond constraint for any element-charge combination is 8
        element-charge not specified in self.bond_constraint will receive the default bond constraint
        if the main key element doesn't follow the string in element_map, the bond constraint will not be implemented in the atom environment but no errors will be given
    additional_charges: dict of list
        {element(str): [charges(int)]}
        additional charges for the elements not included in the default_charges
        all elements from the instantiation will have only [0] as the possible charge even those not specified in self.allowed_charges
        charges already specified in default_charges cannot be removed (as for the reason that it is possible for the element to have that charge)
        the only way to change is to change the whole self.allowed_charges after instantiation
        if charges needs to be masked, change the bond_constraint using additional_bond_contraints

    """

    def __init__(
        self,
        elements: Union[List, str, int] = "organic",
        additional_charges: Optional[dict] = None,
        additional_bond_constraints: Optional[dict] = None,
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
        if len(self.elements) == 0:
            raise ValueError(
                f"elements cannot be empty or should be the atomic symbol in the periodic table"
            )
        # here, we process the additional allowed charges
        # (note that all elements can be uncharged)
        self.allowed_charges = default_charges
        self.bond_constraints = default_bond_constraints
        if additional_charges:
            for element, charges in additional_charges.items():
                self.allowed_charges.setdefault(element, [])
                self.allowed_charges[element] = sorted(
                    set(self.allowed_charges[element] + charges)
                )
        if additional_bond_constraints:
            for element, charges in additional_bond_constraints.items():
                self.bond_constraints.setdefault(element, {}).update(charges)
        self.source_readable = source_readable
        # Source state: [bond, isotope_number   , element, chiral_tag, H_count, charge]
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
        # default constraint is 8
        self.element_to_bond_constraint = {
            element: max(self.bond_constraints.get(element, {0: 8}).values())
            for element in self.action_map["element"]
        }
        # default constraint is 8
        self.charged_element_bond_constraints = {
            self.element_to_action_index[element]: {
                i: bond_constraints.get(element + charge, 8)
                for i, charge in enumerate(self.action_map["charge"])
            }
            for element in self.action_map["element"]
        }
        # create max bond numbers from the initialized element
        self.max_bond_limit = max(
            max(bond_limits.values()) for bond_limits in self.bond_constraints.values()
        )
        self.all_charges = list(
            set(
                charge_i
                for charge_i in self.action_map["charge"]
                for elements in self.action_map["element"]
                if int(charge_i.replace("+", "")) in self.allowed_charges[elements]
            )
        )
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

    def _is_eos_valid(self, state: List[int]) -> bool:
        """
        Checks if the state is valid
        By looking at state
        and looking at if the bonds and charges of the elements are valid
        """
        if not self._is_element_set(state):
            return False
        else:
            # check the element, bond, and state
            # check using the bond constraints
            element = self.action_map["element"][state[2] - 1]
            if self._is_charge_set(state):
                # charge is set
                atom_charge = int(
                    self.action_map["charge"][state[5] - 1].replace("+", "")
                )
                # check if charge is in the allowed charges
                if atom_charge not in self.allowed_charges[element]:
                    # this should be caught but anyway return False
                    return False
            else:
                # atom charge 0 is always allowed
                atom_charge = 0
            # then get the bond
            if self._is_bond_set(state):
                bond_number = self.bond_map[self.action_map["bond"][state[0] - 1]]
                if bond_number <= self.bond_constraints[element][atom_charge]:
                    return True
                else:
                    return False
            else:
                return True

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

    def _is_element_set(self, state: List[int]) -> bool:
        """
        Checks if the element is set
        By looking at state[2]
        If state[2] == 0 then there is no element yet
        else if state[2] != 0 then an element was chosen
        """
        if state[2] != 0:
            return True
        else:
            return False

    def _is_charge_set(self, state: List[int]) -> bool:
        """
        Checks if the charge is set
        By looking at state[6]
        If state[5] == 0 then there is no charge yet
        else if state[5] != 0 then an charge was chosen
        """
        if state[5] != 0:
            return True
        else:
            return False

    def _get_valid_bonds(self, state):
        """
        Create a mask for bonds
        for bonds, we need to check Element and Charge
        (1/4) Element Filled Charge Filled
        (2/4) Element 0 Charge Filled
        (3/4) Element Filled Charge 0
        (4/4) Element 0 Charge 0
        """
        # get number of actions corresponding to bonds
        num_bonds = self.num_actions["bond"]
        valid_bonds = [False] * num_bonds
        # first: (1/4) Element FILLED Charge FILLLED
        if (state[2] != 0) and (state[5] != 0):
            # get the element
            element = self.action_map["element"][state[2] - 1]
            # get the charge
            atom_charge = int(self.action_map["charge"][state[5] - 1].replace("+", ""))
            # if there is a charge, concatenate the element and the charge
            element_bond_constraint = self.bond_constraints[element].get(atom_charge, 0)
            # mask some of the "bond" actions based on the bond constraints
            mask_start = min(3, element_bond_constraint) - 1
            mask_length = num_bonds - min(3, element_bond_constraint) + 1
            # mask all the bonds based on the bond constraints
            valid_bonds[mask_start : mask_start + mask_length] = [True] * mask_length
        # (2/4) Element 0 Charge FILLED
        elif (state[2] == 0) and (state[5] != 0):
            # in this part, we have a charge, so we need to check which elements CAN have that charge
            # the charge in the state
            atom_charge_str = self.action_map["charge"][state[5] - 1]
            atom_charge = abs(int(atom_charge_str))
            # filter the elements with charge constraints less than the chosen charge
            # from the valid elements, get the maximum bond
            max_bond = max(
                [
                    bond_limit_map.get(atom_charge, 0)
                    for element, bond_limit_map in self.bond_constraints.items()
                ]
            )
            mask_start = min(3, max_bond) - 1  # only 2 bonds are allowed
            mask_length = num_bonds - min(3, self.max_bond_limit) + 1
            # mask all the bonds based on the bond constraints
            valid_bonds[mask_start : mask_start + mask_length] = [True] * mask_length
        # (3/4) Element FILLED Charge 0
        elif (state[2] != 0) and (state[5] == 0):  # DONE
            element = self.action_map["element"][state[2] - 1]
            # get the maximum bond limit
            element_bond_constraint = max(self.bond_constraints[element].values())
            if element_bond_constraint < 2:  # only organic elements are considered
                # if the bond_constrain = 1 then the element cannot be  (in a molecule)
                # mask the charges
                valid_bonds = [True] * num_bonds
            else:
                # mask some of the "bond" actions based on the bond constraints
                mask_start = min(3, element_bond_constraint) - 1
                mask_length = num_bonds - min(3, element_bond_constraint) + 1
                # mask all the bonds based on the bond constraints
                valid_bonds[mask_start : mask_start + mask_length] = [
                    True
                ] * mask_length
        # (4/4) Element 0 Charge 0
        else:
            # since there are no element and charge yet
            # we can check maximum bond possible from the environment elements
            # which is the default
            # as a default: we can check all valid bonds based on the environment elements
            mask_start = min(3, self.max_bond_limit) - 1
            mask_length = num_bonds - min(3, self.max_bond_limit) + 1
            # mask all the bonds based on the bond constraints
            valid_bonds[mask_start : mask_start + mask_length] = [True] * mask_length
        return valid_bonds

    # TODO:
    def _get_valid_elements(self, state):
        # get number of actions corresponding to elements
        num_elements = self.num_actions["element"]
        masked_elements = [False] * num_elements
        # for elements, we need to check Bond and Charge
        # (1/4) Bond FILLED Charge FILLED
        if (state[0] != 0) and (state[5] != 0):  # DONE
            # add the bond number to the bond filter
            # since there is already a charge, loop through all the elements and check the bond constraint
            # chosen bond
            bond_number = self.bond_map[self.action_map["bond"][state[0] - 1]]
            # the charge in the state
            atom_charge_str = self.action_map["charge"][state[5] - 1]
            atom_charge = int(atom_charge_str.replace("+", ""))
            invalid_elements = [
                element
                # elements are not in bond_constraints will NEVER be filtered out
                for element, bond_limit_map in self.bond_constraints.items()
                if bond_limit_map.get(atom_charge, 0) < bond_number
            ]
            # filter-out the elements from the possible actions
            for x in invalid_elements:
                masked_elements[self.element_to_action_index[x]] = True
        # (2/4) Bond 0 Charge FILLED
        elif (state[0] == 0) and (state[5] != 0):  # DONE
            # the charge in the state
            atom_charge_str = self.action_map["charge"][state[5] - 1]
            atom_charge = int(atom_charge_str)
            # filter the elements with charge constraints less than the chosen charge
            invalid_elements = [
                element
                for element, bond_limit_map in self.bond_constraints.items()
                if atom_charge not in bond_limit_map.keys()
            ]
            for x in invalid_elements:
                masked_elements[self.element_to_action_index[x]] = True
        # (3/4) Bond FILLED Charge 0
        elif (state[0] != 0) and (state[5] == 0):  # NOT DONE
            # filter actions for some elements
            # first, get the bond
            bond_number = (
                self.bond_map[self.action_map["bond"][state[0] - 1]]
                if state[0] > 0
                else 0
            )
            # invalid elements are elements with bond contraints smaller than the chosen bond number
            invalid_elements = [
                element
                # elements are not in bond_constraints will never be filtered out
                for element, bond_limit in self.element_to_bond_constraint.items()
                if bond_limit < bond_number
            ]
            # loop through all the elements (Use a loop for now)
            for x in invalid_elements:
                masked_elements[self.element_to_action_index[x]] = True
        # (4/4) Bond 0 Charge 0
        else:
            # no filter so use the default
            pass
        return masked_elements

    # TODO:
    def _get_valid_charges(self, state):
        # get number of actions corresponding to charge
        num_charges = self.num_actions["charge"]
        masked_charges = [False] * num_charges
        # for charge, we need to check Bond and Element
        # (1/4) Bond FILLED Element FILLED
        if (state[0] != 0) and (state[2] != 0):
            # first, get the chosen bond
            bond_number = self.bond_map[self.action_map["bond"][state[0] - 1]]
            # for all possible charges get the bond contraint of that by concatenating
            # if the bond constraint is less than bond in state[0] then the mask for that action is invalid
            masked_charges = [
                bond_number
                > self.bond_constraints.get(
                    self.action_map["element"][state[2] - 1], {}
                ).get(int(charge_i.replace("+", "")), 0)
                for charge_i in self.action_map["charge"]
            ]  # False if bond number is over the bond_limit
        # (2/4) Bond 0 Element FILLED
        elif (state[0] == 0) and (state[2] != 0):
            element = self.action_map["element"][state[2] - 1]
            masked_charges = [
                int(charge_i.replace("+", ""))
                not in self.allowed_charges.get(element, [])
                for charge_i in self.action_map["charge"]
            ]
        # (3/4) Bond FILLED Element 0
        elif (state[0] != 0) and (state[2] == 0):
            # first, get the bond
            bond_number = (
                self.bond_map[self.action_map["bond"][state[0] - 1]]
                if state[0] > 0
                else 0
            )
            # consider all the possible elements that can have this bond
            # valid elements are elements with bond contraints smaller than the chosen bond number
            valid_charges = [
                charges
                for bond_limit_map in self.bond_constraints.values()
                for charges, bonds in bond_limit_map.items()
                if bonds > bond_number
            ]
            masked_charges = [
                int(charge_i.replace("+", "")) not in valid_charges
                for charge_i in self.action_map["charge"]
            ]
        # (4/4) Bond 0 Element 0
        else:  # DONE
            masked_charges = [
                charge_i not in self.all_charges
                for charge_i in self.action_map["charge"]
            ]
        return masked_charges

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
        num_elements = self.num_actions["element"]
        num_charges = self.num_actions["charge"]
        # create a default mask for the other actions
        num_isotope_number = self.num_actions["isotope_number"]
        masked_isotopes = [True] * num_isotope_number
        num_chiral_tag = self.num_actions["chiral_tag"]
        masked_chiral_tag = [True] * num_chiral_tag
        num_h_count = self.num_actions["h_count"]
        masked_h_count = [True] * num_h_count

        # for the mask, we limit the action space based on both
        # the action_type and action_value of the action_map
        # Determine which bonds are valid
        if self._is_bond_set(state):
            masked_bonds = [True] * num_bonds
        else:
            masked_bonds = self._get_valid_bonds(state=state)

        # Determine which elements are valid
        if self._is_element_set(state):
            masked_elements = [True] * num_elements
            terminal_mask = [False]
        else:
            masked_elements = self._get_valid_elements(state=state)
            terminal_mask = [True]

        # Determine which charges are valid
        if self._is_charge_set(state):
            masked_charges = [True] * num_charges
        else:
            masked_charges = self._get_valid_charges(state=state)

        if self._is_eos_valid(state):
            terminal_mask = [False]
        else:
            terminal_mask = [True]

        # after checking everything,
        # combine lists to create a mask for the whole action space
        mask = (
            masked_bonds
            + masked_isotopes
            + masked_elements
            + masked_chiral_tag
            + masked_h_count
            + masked_charges
            + terminal_mask
        )
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
            elif bool(
                self.element_to_bond_constraint[state_tuple[2] - 1]
            ) and (  # <<< keys here are wrong
                state_tuple[0] != 0
            ):
                # get the bond constraint
                bond_string = self.action_map["bond"][state_tuple[0] - 1]
                if (
                    self.bond_map[bond_string]
                    <= self.element_to_bond_constraint[
                        state_tuple[2] - 1
                    ]  # <<< keys here are wrong
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

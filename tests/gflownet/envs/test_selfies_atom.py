import warnings

import common
import pytest

from gflownet.envs.selfies.atom import Atom
from gflownet.utils.common import copy


@pytest.fixture
def env_default():
    return Atom()


@pytest.fixture
def env_organic_isotope6_hcount4_charge1():
    return Atom(
        elements="organic",
        allow_chiral_tag=False,
        allow_stereoisomer=True,
        isotope_limit=6,
        h_count_limit=4,
        charge_limit=1,
    )


@pytest.fixture
def env_organic_single_charge():
    return Atom(
        elements="organic",
        allow_chiral_tag=False,
        allow_stereoisomer=False,
        isotope_limit=0,
        h_count_limit=0,
        charge_limit=1,
    )


@pytest.fixture
def env_organic_charge4():
    return Atom(
        elements="organic",
        allow_chiral_tag=False,
        allow_stereoisomer=False,
        isotope_limit=0,
        h_count_limit=0,
        charge_limit=4,
    )


@pytest.fixture
def env_source_readable_notdefault():
    return Atom(source_readable="<source-test>")


@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_source_readable_notdefault",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


DEFAULT_ACTIONS = [
    (1, 1),
    (1, 2),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (6, 1),
    (6, 2),
    (-1, -1),
]
# action to states mapping
DEFAULT_ORGANIC_ELEMENTS = {
    "H": 1,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "F": 6,
    "P": 7,
    "S": 8,
    "Cl": 9,
    "Br": 10,
    "I": 11,
}
DEFAULT_CHARGES = {"-1": 1, "+1": 2}
DEFAULT_BONDS = {"=": 1, "#": 2}


# states are the indices of the atoms
@pytest.mark.parametrize(
    "env, action_space",
    [
        (
            "env_organic_isotope6_hcount4_charge1",
            [
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (3, 7),
                (3, 8),
                (3, 9),
                (3, 10),
                (3, 11),
                (5, 1),
                (5, 2),
                (5, 3),
                (5, 4),
                (6, 1),
                (6, 2),
                (-1, -1),
            ],
        )
    ],
)
def test__get_action_space__returns_expected(env, action_space, request):
    env = request.getfixturevalue(env)
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_exp, parent_actions_exp",
    [
        (
            [2, 0, 7, 0, 0, 0],  # [#P]
            [[0, 0, 7, 0, 0, 0], [2, 0, 0, 0, 0, 0]],
            [
                (
                    1,
                    2,
                ),
                (
                    3,
                    7,
                ),
            ],
        ),
        (
            [1, 0, 4, 0, 0, 0],  # [=N]
            [[0, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
            [
                (
                    1,
                    1,
                ),
                (
                    3,
                    4,
                ),
            ],
        ),
        (
            [1, 0, 0, 0, 0, 0],  # [=]
            [[0, 0, 0, 0, 0, 0]],
            [
                (
                    1,
                    1,
                )
            ],
        ),
        (
            [0, 0, 4, 0, 0, 0],  # [N]
            [[0, 0, 0, 0, 0, 0]],
            [
                (
                    3,
                    4,
                )
            ],
        ),
    ],
)
def test__get_parents__returns_expected(
    env_organic_single_charge, state, parents_exp, parent_actions_exp
):
    env = env_organic_single_charge

    parents, parent_actions = env.get_parents(state)

    # Create dictionaries of parent_action: parent for comparison
    parents_actions_exp_dict = {}
    for parent, action in zip(parents_exp, parent_actions_exp):
        parents_actions_exp_dict[action] = tuple(parent.copy())
    parents_actions_dict = {}
    for parent, action in zip(parents, parent_actions):
        parents_actions_dict[action] = tuple(parent.copy())

    # Compare sorted dictionaries
    assert sorted(parents_actions_exp_dict) == sorted(parents_actions_dict)


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [2, 0, 7, 0, 0, 0],
            (6, 1),
            [2, 0, 7, 0, 0, 1],
        ),
        (
            [0, 0, 0, 0, 0, 0],
            (3, 1),
            [0, 0, 1, 0, 0, 0],
        ),
        (
            [0, 0, 1, 0, 0, 0],
            (1, 1),  # invalid action
            [0, 0, 1, 0, 0, 0],
        ),
        (
            [0, 0, 1, 0, 0, 0],
            (6, 1),  # invalid action
            [0, 0, 1, 0, 0, 0],
        ),
        (
            [0, 0, 0, 0, 0, 1],  # failed
            (3, 1),  # invalid action
            [0, 0, 0, 0, 0, 1],
        ),
        (
            [2, 0, 7, 0, 0, 0],
            (-1, -1),  # terminating action
            [2, 0, 7, 0, 0, 0],
        ),
    ],
)
def test__step__returns_expected(env_organic_single_charge, state, action, next_state):
    env = env_organic_single_charge
    env.set_state(state)
    env.step(action)
    assert env.state == next_state


# tests for masks
@pytest.mark.parametrize(
    "env, state, mask_expected",
    [
        # Source state
        (
            "env_organic_charge4",
            [2, 0, 0, 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (6, 3): True,
                    (6, 4): False,
                    (6, 5): False,
                    (6, 6): True,
                    (6, 7): True,
                    (6, 8): True,
                    (-1, -1): True,
                }.values()
            ),
        ),  # <<< This is not yet passed because it needs to change #O as NOT terminal
        (
            "env_organic_charge4",
            [2, 0, 5, 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (6, 3): True,
                    (6, 4): True,
                    (6, 5): False,
                    (6, 6): True,
                    (6, 7): True,
                    (6, 8): True,
                    (-1, -1): True,
                }.values()
            ),
        ),  # <<< This is not yet passed because it needs to change #O as NOT terminal
        (
            "env_organic_charge4",
            [0, 0, 0, 0, 0, 0],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): False,
                    (6, 1): True,
                    (6, 2): True,
                    (6, 3): True,
                    (6, 4): False,
                    (6, 5): False,
                    (6, 6): True,
                    (6, 7): True,
                    (6, 8): True,
                    (-1, -1): True,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, 1, 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, 0, 0, 0, 1],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): True,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, 0, 0, 0, 2],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): True,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, 1, 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [1, 0, 0, 0, 0, 1],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): True,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [DEFAULT_BONDS["="], 0, 0, 0, 0, DEFAULT_CHARGES["+1"]],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["H"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["B"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["C"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["N"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["O"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["F"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["P"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["S"]): False,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Cl"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["Br"]): True,
                    (3, DEFAULT_ORGANIC_ELEMENTS["I"]): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): True,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, 0, 0, 0, 0],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, 1): False,
                    (3, 2): False,
                    (3, 3): False,
                    (3, 4): False,
                    (3, 5): False,
                    (3, 6): False,
                    (3, 7): False,
                    (3, 8): False,
                    (3, 9): False,
                    (3, 10): False,
                    (3, 11): False,
                    (6, 1): False,
                    (6, 2): False,
                    (-1, -1): True,
                }.values()
            ),
        ),
        # Option selected
        (
            "env_organic_single_charge",
            [0, 0, DEFAULT_ORGANIC_ELEMENTS["B"], 0, 0, DEFAULT_CHARGES["-1"]],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, 1): True,
                    (3, 2): True,
                    (3, 3): True,
                    (3, 4): True,
                    (3, 5): True,
                    (3, 6): True,
                    (3, 7): True,
                    (3, 8): True,
                    (3, 9): True,
                    (3, 10): True,
                    (3, 11): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, DEFAULT_ORGANIC_ELEMENTS["B"], 0, 0, DEFAULT_CHARGES["+1"]],
            list(
                {
                    (1, 1): False,
                    (1, 2): True,
                    (3, 1): True,
                    (3, 2): True,
                    (3, 3): True,
                    (3, 4): True,
                    (3, 5): True,
                    (3, 6): True,
                    (3, 7): True,
                    (3, 8): True,
                    (3, 9): True,
                    (3, 10): True,
                    (3, 11): True,
                    (6, 1): True,
                    (6, 2): True,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [0, 0, DEFAULT_ORGANIC_ELEMENTS["B"], 0, 0, 0],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (3, 1): True,
                    (3, 2): True,
                    (3, 3): True,
                    (3, 4): True,
                    (3, 5): True,
                    (3, 6): True,
                    (3, 7): True,
                    (3, 8): True,
                    (3, 9): True,
                    (3, 10): True,
                    (3, 11): True,
                    (6, 1): False,
                    (6, 2): False,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [DEFAULT_BONDS["="], 0, DEFAULT_ORGANIC_ELEMENTS["B"], 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, 1): True,
                    (3, 2): True,
                    (3, 3): True,
                    (3, 4): True,
                    (3, 5): True,
                    (3, 6): True,
                    (3, 7): True,
                    (3, 8): True,
                    (3, 9): True,
                    (3, 10): True,
                    (3, 11): True,
                    (6, 1): False,
                    (6, 2): False,
                    (-1, -1): False,
                }.values()
            ),
        ),
        (
            "env_organic_single_charge",
            [DEFAULT_BONDS["#"], 0, DEFAULT_ORGANIC_ELEMENTS["B"], 0, 0, 0],
            list(
                {
                    (1, 1): True,
                    (1, 2): True,
                    (3, 1): True,
                    (3, 2): True,
                    (3, 3): True,
                    (3, 4): True,
                    (3, 5): True,
                    (3, 6): True,
                    (3, 7): True,
                    (3, 8): True,
                    (3, 9): True,
                    (3, 10): True,
                    (3, 11): True,
                    (6, 1): False,
                    (6, 2): True,
                    (-1, -1): False,
                }.values()
            ),
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(
    env, state, mask_expected, request
):
    env = request.getfixturevalue(env)
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_expected
    env.set_state(state)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_expected


# Adapt this test to make it reflect the expected behaviour. It seems that this is
# correct and instead the forward mask may be wrong.
@pytest.mark.skip(reason="skip until it gets updated")
def test__get_all_terminating_states_valid(env_default):
    """
    inputs:
        env_default: Atom() env
        terminating_state: state (list)
    """
    # restart env
    env = env_default
    terminating_states = env.get_all_terminating_states()
    for terminating_state in terminating_states:
        env.set_state(terminating_state)
        # 1. check that the mask is False for the terminal
        mask = env.get_mask_invalid_actions_forward(terminating_state, done=False)
        assert mask[-1] == False  # then it is really terminating
        parents, parents_a = env.get_parents()
        # 2. set that the parent can take an action_expected to reach terminating_state
        # do this to all parents
        for p, p_a in zip(parents, parents_a):
            env.set_state(p)
            valid_actions = env.get_valid_actions(done=False)
            # check that the action is in the valid actions
            assert p_a in valid_actions


@pytest.mark.repeat(5)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_source_readable_notdefault",
        "env_organic_single_charge",
        "env_organic_isotope6_hcount4_charge1",
        "env_organic_charge4",
    ],
)
def test__step_random__does_not_crash_from_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    state_next, action, valid = env.step_random()
    assert True


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_source_readable_notdefault",
        "env_organic_single_charge",
        "env_organic_isotope6_hcount4_charge1",
        "env_organic_charge4",
    ],
)
def test__step_random__does_not_get_stuck_in_no_valid_actions(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    states = [copy(env.state)]
    actions = []
    while not env.done:
        mask = env.get_mask_invalid_actions_forward()
        if all(mask):
            warnings.warn("IMPORTANT: Found invalid action!")
            import pdb

            pdb.set_trace()
            assert False
        assert not all(mask)
        state_next, action, valid = env.step_random()
        if valid:
            states.append(state_next)
            actions.append(action)
    assert True


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_source_readable_notdefault",
        "env_organic_single_charge",
        "env_organic_isotope6_hcount4_charge1",
        "env_organic_charge4",
    ],
)
def test__step_random__does_not_crash_and_reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    states = [copy(env.state)]
    actions = []
    while not env.done:
        state_next, action, valid = env.step_random()
        if valid:
            states.append(copy(state_next))
            actions.append(action)
        else:
            warnings.warn("IMPORTANT: Found invalid action!")
    assert True


# TestAtomBasic
# TestAtomOrganicCharge4
# TestAtomOrganicSingleCharge


class TestAtomBasic(common.BaseTestsDiscrete):
    """Common tests for default Atom (organic) environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_default):
        self.env = env_default
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestAtomOrganicSingleCharge(common.BaseTestsDiscrete):
    """
    Common tests for default Atom (organic) environment
    parameter values:
        elements = "organic"
        allow_chiral_tag = False
        allow_stereoisomer = False
        isotope_limit = 0
        h_count_limit = 0
        charge_limit = 1
    """

    @pytest.fixture(autouse=True)
    def setup(self, env_organic_single_charge):
        self.env = env_organic_single_charge
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestAtomOrganicCharge4(common.BaseTestsDiscrete):
    """
    Common tests for Atom (organic) environment
    with env_organic_charge4 environment
    parameters:
        elements = "organic"
        allow_chiral_tag = False
        allow_stereoisomer = False
        isotope_limit = 0
        h_count_limit = 0
        charge_limit = 4
    """

    @pytest.fixture(autouse=True)
    def setup(self, env_organic_charge4):
        self.env = env_organic_charge4
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }

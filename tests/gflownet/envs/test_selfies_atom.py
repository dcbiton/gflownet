import common
import pytest

from gflownet.envs.selfies.atom import Atom


@pytest.fixture
def env_default():
    return Atom()


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
    "action_space",
    [
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
    ],
)
def test__get_action_space__returns_expected(env_default, action_space):
    env = env_default
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
    env_default, state, parents_exp, parent_actions_exp
):
    env = env_default

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
            [0, 0, 0, 0, 0, 1],
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
def test__step__returns_expected(env_default, state, action, next_state):
    env_default.set_state(state)
    env_default.step(action)
    assert env_default.state == next_state


# tests for masks
@pytest.mark.parametrize(
    "state, mask_expected",
    [
        # Source state
        (
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
        # # Option selected
        (
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
    env_default, state, mask_expected
):
    mask = env_default.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_expected
    env_default.set_state(state)
    mask = env_default.get_mask_invalid_actions_forward()
    assert mask == mask_expected


# Adapt this test to make it reflect the expected behaviour. It seems that this is
# correct and instead the forward mask may be wrong.
@pytest.mark.parametrize(
    "state, parent_unexpected, action_unexpected",
    [
        (
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            (3, 1),
        ),
    ],
)
def test__get_parents__does_not_contain_unexpected_parents(
    env_default, state, parent_unexpected, action_unexpected
):
    env = env_default
    env.set_state(state)
    parents, parents_a = env.get_parents()
    for p, p_a in zip(parents, parents_a):
        assert not env.equal(p, parent_unexpected)
        assert p_a != action_unexpected


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

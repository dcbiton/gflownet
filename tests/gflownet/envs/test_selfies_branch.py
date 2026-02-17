import common
import pytest

from gflownet.envs.selfies.branch import Branch


@pytest.fixture
def env():
    return Branch()


def test__environment_initializes_properly():
    env = Branch()
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [
            # fmt: off
            (1,), (2,), (3,), (4,), (5,),(6,), (-1,)
            # fmt: on
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    a = env.get_action_space()
    print(a)
    print(env.action_space_dim)
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [5, 1],
            [[5]],
            [(1,)],
        ),
        (
            [6],
            [[]],
            [(6,)],
        ),
        (
            [1, 5],
            [[1]],
            [(5,)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected
):
    parents, parents_a = env.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert p == p_e
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [],
            (1,),
            [1],
        ),
        (
            [1],
            (3,),
            [1],
        ),
        (
            [5],
            (2,),
            [5, 2],
        ),
        (
            [1],
            (1,),
            [1],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(state)
    env.step(action)
    assert env.state == next_state


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0, 0, 0], ""),
        ([3, 1, 20, 0, 0, 0, 0], "C A T"),
        ([2, 9, 18, 4, 19, 0, 0], "B I R D S"),
        ([13, 1, 3, 8, 9, 14, 5], "M A C H I N E"),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0, 0, 0], ""),
        ([3, 1, 20, 0, 0, 0, 0], "C A T"),
        ([2, 9, 18, 4, 19, 0, 0], "B I R D S"),
        ([13, 1, 3, 8, 9, 14, 5], "M A C H I N E"),
    ],
)
def test__readable2state__returns_expected(env, state, readable):
    assert env.readable2state(readable) == state


class TestBranchBasic(common.BaseTestsDiscrete):
    """Common tests for default Branch environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
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

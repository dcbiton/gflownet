import common
import pytest

from gflownet.envs.selfies.branch import Branch


@pytest.fixture
def env():
    return Branch()


def test__environment_initializes_properly():
    env = Branch()
    assert True


# states are the indices of the atoms
@pytest.mark.parametrize(
    "action_space",
    [
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (-1, -1),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_exp, parent_actions_exp",
    [
        (
            [1, 1],  # [=Branch1]
            [[1, 0], [0, 1]],
            [
                (
                    2,
                    1,
                ),
                (
                    1,
                    1,
                ),
            ],
        ),
        (
            [2, 1],  # [#Branch1]
            [[2, 0], [0, 1]],
            [
                (
                    2,
                    1,
                ),
                (
                    1,
                    2,
                ),
            ],
        ),
        (
            [1, 2],  # [=Branch2]
            [[1, 0], [0, 2]],
            [
                (
                    2,
                    2,
                ),
                (
                    1,
                    1,
                ),
            ],
        ),
        (
            [2, 2],  # [#Branch2]
            [[2, 0], [0, 2]],
            [
                (
                    2,
                    2,
                ),
                (
                    1,
                    2,
                ),
            ],
        ),
        (
            [1, 3],  # [=Branch3]
            [[1, 0], [0, 3]],
            [
                (
                    2,
                    3,
                ),
                (
                    1,
                    1,
                ),
            ],
        ),
        (
            [2, 3],  # [#Branch3]
            [[2, 0], [0, 3]],
            [
                (
                    2,
                    3,
                ),
                (
                    1,
                    2,
                ),
            ],
        ),
        (
            [0, 1],  # [Branch1]
            [[0, 0]],
            [
                (
                    2,
                    1,
                )
            ],
        ),
        (
            [0, 2],  # [Branch2]
            [[0, 0]],
            [
                (
                    2,
                    2,
                )
            ],
        ),
        (
            [0, 3],  # [Branch3]
            [[0, 0]],
            [
                (
                    2,
                    3,
                )
            ],
        ),
        (
            [1, 0],  # [=Branch]
            [[0, 0]],
            [
                (
                    1,
                    1,
                )
            ],
        ),
        (
            [2, 0],  # [#Branch]
            [[0, 0]],
            [
                (
                    1,
                    2,
                )
            ],
        ),
    ],
)
def test__get_parents__returns_expected(env, state, parents_exp, parent_actions_exp):
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
            [2, 0],
            (2, 1),
            [2, 1],
        ),
        (
            [0, 0],
            (2, 1),
            [0, 1],
        ),
        (
            [0, 2],
            (1, 1),
            [1, 2],
        ),
        (
            [0, 3],
            (1, 1),
            [1, 3],
        ),
        (
            [0, 0],
            (2, 3),
            [0, 3],
        ),
        (
            [0, 0],
            (-1, -1),
            [0, 0],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(state)
    env.step(action)
    assert env.state == next_state


# tests for masks
@pytest.mark.parametrize(
    "state, mask_expected",
    [
        # Source state
        (
            [0, 0],
            list(
                {
                    (1, 1): False,
                    (1, 2): False,
                    (2, 1): False,
                    (2, 2): False,
                    (2, 3): False,
                    (-1, -1): True,
                }.values()
            ),
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(
    env, state, mask_expected, request
):
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_expected
    env.set_state(state)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_expected


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

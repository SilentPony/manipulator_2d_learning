#!/usr/bin/env python3
#coding:utf-8
"""
A reinforcement learning problem of a simple model of a 2D manipulator that must reach
a goal point with the end effector by sending speed commands to joints (setting discrete
angle translations).
"""

from typing import Optional, Tuple, Any

import signal

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import gym
import gym.wrappers


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.

    Code adapted from gym.seeding > 0.21.

    Args:
        seed: The seed used to create the generator
    Returns:
        The generator and resulting seed
    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise TypeError(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise ValueError(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


def goal_distance(goal_a, goal_b):
    """
    Calculated distance between two goal poses (usually an achieved pose
    and a required pose).
    """
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DiscreteActionFlatManipulatorEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        distance_threshold = 0.3,
        target_range = 1.0,
        initial_pose = None,
        randomize_initial_pose = False,
        min_degree_angle = -160.,
        max_degree_angle = 160.,
        min_degree_angle_speed = -5.,
        max_degree_angle_speed = 5.,
        angle_step_degree = 1.,
        randomize_goal = True,
        static_goal = None,
        reward_type = "sparse",
        render_mode = None,
        seed = None,
        debug = False
    ):
        super().__init__()

        self.debug = debug
        self.render_mode = render_mode

        # pyplot figure for plotting
        self.fig = None

        self.seed = seed
        self.rng, self.rng_seed = np_random(seed)

        self.target_range = target_range
        self.reward_type = reward_type

        self.randomize_initial_pose = randomize_initial_pose
        self.initial_pose = initial_pose

        # Допустимая разница между реальным положением конца звена и целью
        self.distance_threshold = distance_threshold

        # manipulator geometry
        self.link1_len = 1.  # meters
        self.link2_len = 1.

        # limits from degrees to radians
        self.min_angle = np.radians(min_degree_angle)
        self.max_angle = np.radians(max_degree_angle)

        self.min_angle_speed = np.radians(min_degree_angle_speed)
        self.max_angle_speed = np.radians(max_degree_angle_speed)

        # joints (from base to end effector)
        self.joints = np.copy(self.initial_pose)

        # link ends (from base to end effector)
        self.link1_end, self.link2_end = self._calc_link_ends()

        angle_step = np.radians(angle_step_degree)
        possible_actions_1d = np.arange(
            self.min_angle_speed, self.max_angle_speed,
            angle_step, dtype="float32"
        )

        # cartesian product of arrays: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-111.php
        self.possible_actions_2d = np.transpose([
            np.tile(possible_actions_1d, len(possible_actions_1d)),
            np.repeat(possible_actions_1d, len(possible_actions_1d))
        ])

        self.randomize_goal = randomize_goal
        self.static_goal = static_goal
        self.goal = np.zeros(0)

        obs = self._get_obs()

        self.action_space = gym.spaces.Discrete(len(self.possible_actions_2d))
        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=gym.spaces.Box(
                    low=np.array([-1., -1.], dtype="float32"),
                    high=np.array([1., 1.], dtype="float32"),
                    shape=obs["achieved_goal"].shape,
                    dtype="float32",
                ),

                achieved_goal=gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype="float32",
                ),

                observation=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

        # extra info
        self.iteration = 0

    def _sim_calc(self):
        """
        Updates simulation state according to various internal parameters (like joints)
        """
        # if an action moved joints outside of allowed, clip it without any reward punishment
        # which corresponds to a physical safety limiter (or a low level control software cutoff)
        self.joints = np.clip(self.joints, self.min_angle, self.max_angle)

        # update kinematics
        self._calc_link_ends()

    def _sim_step(self, action_angles):
        self.joints = np.add(self.joints, action_angles)

        self._sim_calc()

    def _calc_link_ends(self):
        # link1
        angle1 = self.joints[0]
        rot_mat = np.array(
            [[np.cos(angle1), -np.sin(angle1)],
             [np.sin(angle1), np.cos(angle1)]],
            dtype="float32"
        )

        link1 = np.array([self.link1_len, 0.], dtype="float32")
        self.link1_end = rot_mat.dot(link1)

        # link2
        angle2 = self.joints[0] + self.joints[1]
        rot_mat = np.array(
            [[np.cos(angle2), -np.sin(angle2)],
             [np.sin(angle2), np.cos(angle2)]],
            dtype="float32"
        )

        link2 = np.array([self.link2_len, 0.], dtype="float32")
        self.link2_end = self.link1_end + rot_mat.dot(link2)

        return self.link1_end, self.link2_end

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _get_obs(self):
        obs = np.array([
            self.joints.ravel(),
            self.link1_end.ravel(),
            self.link2_end.ravel()
        ], dtype="float32")

        achieved_goal = np.array(self.link2_end.ravel(), dtype="float32")

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)

        return (d < self.distance_threshold).astype(np.float32)

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Action {action} not from allowed set {self.action_space}")

        action_angles = self.possible_actions_2d[action]

        self._sim_step(action_angles)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = float(self.compute_reward(obs["achieved_goal"], self.goal, info))

        # the goal of the environment is to continuously achieve the desired pose
        terminated = False

        # extra info update
        self.iteration += 1

        return obs, reward, terminated, info

    def _sample_goal(self):
        goal = self.rng.uniform(
            -self.target_range, self.target_range, size=2
        ).astype("float32")

        return goal.copy()

    def reset(self):
        super().reset()

        # reset initial state
        if self.randomize_initial_pose or (self.initial_pose is None):
            self.initial_pose = self.rng.uniform(
                self.min_angle, self.max_angle, size=2
            ).astype("float32")

        self.joints[:] = np.copy(self.initial_pose)
        self._sim_calc()

        # get a new random goal if randomization or if no goal specified
        if self.randomize_goal or (self.goal.size == 0 and self.static_goal is None):
            self.goal = self._sample_goal().copy()

        if (not self.randomize_goal) and (self.static_goal is not None):
            self.goal = np.array(self.static_goal, dtype="float32")

        # get the state after reset
        obs = self._get_obs()

        # reset extra info
        self.iteration = 0

        if self.render_mode == "human":
            self.render()

        return obs

    def render(self, mode='human'):
        with np.printoptions(precision=2, suppress=True):
            print(
                f'Iteration: {self.iteration}\nGoal: {self.goal}\n'
                f'End effector: {self.link2_end.ravel()}\nJoints: {self.joints.ravel()}\n'
            )

        self.plot()

    def plot(self):
        plt.ion()
        if self.fig is None:
            self.fig = plt.figure()
        plt.clf()

        ax = self.fig.add_subplot(111)

        plt.title("Manipulator and goal")
        plt.xlim([-5, 5])
        plt.ylim(-5, 5)
        plt.grid()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

        # plot goal point
        ax.plot(self.goal[0],self.goal[1],'-*', label = 'Goal pose')

        # plot manipulator links
        link_x = [0, self.link1_end[0], self.link2_end[0]]
        link_y = [0, self.link1_end[1], self.link2_end[1]]
        ax.plot(link_x, link_y, '-*', label = 'Current pose')

        plt.legend()
        plt.show()
        plt.pause(0.2)

    def close (self):
        pass


def make_env(max_episode_len=None, render_mode=None):
    """
    Creates a wrapped environment.
    """
    env = DiscreteActionFlatManipulatorEnv(
        randomize_initial_pose=True,
        initial_pose=(0., 0.),
        randomize_goal=True,
        static_goal=(1., 1.),
        render_mode=render_mode
    )

    # TimeLimit for efficient sampling
    if max_episode_len:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)

    return env


def test_simple():
    """
    Sends random actions to the environment
    """
    env = make_env(render_mode = 'human')

    obs = env.reset()
    i = 0
    while i < 1000:
        i += 1
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)

        if i % 60 == 0:
            env.reset()
            i = 0


def test_simple_forward():
    env = make_env(render_mode = 'human')

    obs = env.reset()
    i = 0
    while True:
        i += 1
        action = 1
        obs, reward, terminated, info = env.step(action)

        if i % 60 == 0:
            env.reset()
            i = 0


def test_dqn_model(model_name):
    env = make_env(render_mode = 'human')

    model = DQN.load("../models/2d_manipulator_model_GoalEnv/" + model_name, env=env)
    obs = env.reset()

    # number of steps to show after reaching the goal
    success_demo_delay = 5
    success_reached_step = None
    stop = False
    i = 0
    while True:
        i += 1
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, info = env.step(action)

        if info['is_success'] and success_reached_step is None:
            success_reached_step = i

        if success_reached_step is not None:
            if i - success_reached_step > success_demo_delay:
                stop = True

        if terminated or i == 200:
            stop = True

        if stop:
            env.reset()
            i = 0
            stop = False
            success_reached_step = None


def train_dqn_her(num_timesteps, model_name=None):
    if model_name is None:
        model_name = "manipulator_DQN_HER"

    MAX_EPISODE_LEN = 120
    env = make_env(max_episode_len=MAX_EPISODE_LEN)

    obs = env.reset()
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.001,  # 0.0001
        buffer_size=int(1e5), # 1e6
        learning_starts=256, # 2048
        batch_size=256, # 2048
        tau=0.05, # 1.0
        gamma=0.95,
        train_freq=(MAX_EPISODE_LEN, 'step'),
        gradient_steps=1,
        optimize_memory_usage=False,
        target_update_interval=1000, # 10000
        exploration_fraction=0.1, # 0.1
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        seed=None,
        device='auto',
        tensorboard_log="../logs/logs_2d_manipulator_GoalEnv/" + model_name,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            goal_selection_strategy="episode",
            n_sampled_goal=4,
            max_episode_length=MAX_EPISODE_LEN,
            online_sampling=True,
            handle_timeout_termination=True
        ),
        verbose=0,
    )

    # Save a checkpoint regularly
    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e4),
        save_path="../models/2d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=num_timesteps,
        tb_log_name=model_name,
        callback=checkpoint_callback
    )
    model.save("../models/2d_manipulator_model_GoalEnv/" + model_name)


def train_dqn(num_timesteps, model_name=None):
    if model_name is None:
        model_name = "manipulator_DQN"

    MAX_EPISODE_LEN = 120
    env = make_vec_env(
        make_env,
        n_envs=8,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(
            max_episode_len=MAX_EPISODE_LEN
        )
    )

    obs = env.reset()
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.001,  # 0.0001
        buffer_size=int(1e5), # 1e6
        learning_starts=256,
        batch_size=256,
        tau=0.05, # 1.0
        gamma=0.95,
        train_freq=(MAX_EPISODE_LEN, 'step'),
        gradient_steps=1,
        optimize_memory_usage=False,
        target_update_interval=1000, # 10000
        exploration_fraction=0.1, # 0.1
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        seed=None,
        device='auto',
        tensorboard_log="../logs/logs_2d_manipulator_GoalEnv/" + model_name,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        verbose=0,
    )

    # Save a checkpoint regularly
    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e4),
        save_path="../models/2d_manipulator_model_GoalEnv/" + model_name + "/checkpoints/",
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=num_timesteps,
        tb_log_name=model_name,
        callback=checkpoint_callback
    )
    model.save("../models/2d_manipulator_model_GoalEnv/" + model_name)


def main():
    # signal magic to make matplotlib quit on Ctrl+C more reliably
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # timesteps to train
    num_timesteps = int(1e7)

    #test_simple()
    #test_simple_forward()

    #test_dqn_model("random_start_fixed_finish/manipulator_DQN_HER_1e7")
    #test_dqn_model("random_start_random_finish/manipulator_DQN")
    test_dqn_model("random_start_random_finish/manipulator_DQN_HER")

    #train_dqn(num_timesteps)
    #train_dqn_her(num_timesteps)


if __name__ == '__main__':
    main()

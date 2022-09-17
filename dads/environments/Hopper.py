import gym
from soft_actor_critic.environments.environment import Environment
import numpy as np
from gym.spaces import Box


class HopperEnv(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('Hopper-v3')
        self.observation_space = self.env.observation_space
        # self.observation_space = Box(-float('inf'), float('inf'), shape=(27,))
        self.action_space = self.env.action_space

    def take_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.reward = reward
        self.done = done
        self.frames += 1
        return obs, reward, done, _

    def reset_env(self):
        self.observation = self.env.reset()
        self.reward = None
        self.done = False
        self.frames = 0
        return self.observation

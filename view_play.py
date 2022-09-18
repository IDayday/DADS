import gym
import numpy as np
from dads.agent.DADSAgent import DADSAgent
from dads.environments.ant_truncated import Ant_Truncated_State
from dads.environments.HalfCheetah import HalfCheetahEnv
from dads.environments.Hopper import HopperEnv
# from mujoco_py import GlfwContext
import cv2
import numpy as np
import os
import torch

# GlfwContext(offscreen=True)

class Play:
    def __init__(self, env, agent, video_path):
        self.env = env
        self.agent = agent
        self.video_path = video_path
        self.n_skills = agent.n_skills
        self.agent.skill_dynamics.eval()
        self.agent.actor.eval()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') #  'XVID'输出avi 
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') #  'mp4v'输出mp4
        self.device = self.agent.device

        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)

    def evaluate(self):
        
        for z in range(self.n_skills):
            self.env.reset_env()
            self.agent._sample_skill(z)  # one hot encoding
            video_writer = cv2.VideoWriter(self.video_path + f"/skill{z}" + ".avi", self.fourcc, 50.0, (400, 400))
            env_timesteps = 0
            episode_reward = 0
            while not self.env.done or (env_timesteps < 250):
                env_timesteps += 1
                # Take the observation from the environment, format it, push it to GPU
                current_obs = torch.tensor(self.env.observation, dtype=torch.float, device=self.device, requires_grad=False).reshape((1, -1))
                # current_xy_coords = torch.tensor(self.env.xy_coords, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, 2)
                current_obs_skill = torch.cat((current_obs, self.agent.active_skill), 1)
                current_action = self.agent.choose_action(current_obs_skill)
                self.env.take_action(current_action.squeeze().cpu().numpy())
                # next_obs = torch.tensor(self.env.observation, dtype=torch.float, device=self.device, requires_grad=False).reshape((1, -1))
                # done = torch.tensor([[self.env.done]], dtype=torch.int, device=self.device, requires_grad=False)
                reward = self.env.reward
                # next_xy_coords = torch.tensor(self.env.xy_coords, dtype=torch.float, device=self.device, requires_grad=False).reshape(1,2)
                episode_reward += reward
                I = self.env.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (400, 400))
                video_writer.write(I)
            print(f"skill: {z}, episode reward:{episode_reward:.1f}, episode step:{env_timesteps:d}")
        
        self.env.env.close()
        cv2.destroyAllWindows()

def main():
    # env = Ant_Truncated_State()
    # env = HalfCheetahEnv()
    env = HopperEnv()
    device = torch.device("cuda")

    model_path = "./checkpoints/Hopper/09-17 16-34-44/params.pth"
    # save_path = "checkpoints/Ant" + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
    # save_path = "checkpoints/HalfCheetah" + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
    video_path = "./checkpoints/Hopper/09-17 16-34-44/video"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if "Ant" in video_path:
        pos_shape = 2
    else:
        pos_shape = env.env.sim.data.body_xpos.size
    agent = DADSAgent(env=env, device=device, n_skills=20, learning_rate=3e-4, train=False, pos_shape=pos_shape)
    agent.load_models(load_path=model_path)
    player = Play(env, agent, video_path)
    player.evaluate()


if __name__ == "__main__":
    main()


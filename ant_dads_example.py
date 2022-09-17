from imageio import save
import torch
from dads.agent.DADSAgent import DADSAgent
from dads.agent.SkillDynamics import SkillDynamics
from soft_actor_critic.agent.Actor import Actor
from soft_actor_critic.agent.Critic import Critic
from torch import optim
from dads.environments.ant_truncated import Ant_Truncated_State
from dads.environments.HalfCheetah import HalfCheetahEnv
from dads.environments.Hopper import HopperEnv
from datetime import datetime
import time
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# env = Ant_Truncated_State()
# env = HalfCheetahEnv()
env = HopperEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# save_path = "checkpoints/Ant" + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
# save_path = "checkpoints/HalfCheetah" + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
save_path = "checkpoints/Hopper" + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_dir = save_path + "/train.log"
if "Ant" in save_path:
    pos_shape = 2
else:
    pos_shape = env.env.sim.data.body_xpos.size
agent = DADSAgent(env=env, device=device, n_skills=20, learning_rate=3e-4, log_dir=log_dir, pos_shape=pos_shape)

t = 0
while True:
    t += 1
    agent.play_games(1, verbose=False)
    agent.save_models(save_path)
    with open(save_path + "/iterations.txt", "a") as iter_file:
        iter_file.seek(0)
        iter_file.write("iter {} at time: {}\n".format(t, datetime.now(tz=None)))


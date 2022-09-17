import torch
from dads.agent.DADSAgent import DADSAgent
from dads.agent.SkillDynamics import SkillDynamics
from soft_actor_critic.agent.Actor import Actor
from soft_actor_critic.agent.Critic import Critic
from torch import optim
from dads.environments.ant_truncated import Ant_Truncated_State
from datetime import datetime
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

env = Ant_Truncated_State()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DADSAgent(env=env, device=device, n_skills=4, learning_rate=3e-4)

# agent.load_models()

t = 0
while True:
    t += 1
    agent.play_games(1, verbose=False)
    agent.save_models()
    with open("iterations.txt", "a") as iter_file:
        iter_file.seek(0)
        iter_file.write("iter {} at time: {}\n".format(t, datetime.now(tz=None)))

# Display each skill's behaviour for diagnostic purposes:
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=0)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=1)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=2)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=3)


import torch
from dads.agent.DADSAgent import DADSAgent
from dads.agent.SkillDynamics import SkillDynamics
from soft_actor_critic.agent.Actor import Actor
from soft_actor_critic.agent.Critic import Critic
from torch import optim
from dads.environments.ant_truncated import Ant_Truncated_State
from datetime import datetime

env = Ant_Truncated_State()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DADSAgent(env=env, device=device, n_skills=4, learning_rate=3e-4)

for e in range(1000):
    obs = env.reset_env()
    for s in range(1100):
        

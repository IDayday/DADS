import torch
from soft_actor_critic.agent.Memory import Memory

class SkillDynamicsMemory(Memory):
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        super(SkillDynamicsMemory, self).__init__(memory_length=self.memory_length, device=self.device)
        self.skills = torch.tensor([], dtype=torch.int, device=self.device, requires_grad=False)
        self.current_pos = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=False)
        self.next_pos = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=False)

    def sample_memory(self):
        current_pos = self.current_pos
        next_pos = self.next_pos
        skills = self.skills
        obs = self.observation
        next_obs = self.next_observation
        actions = self.action
        rewards = self.reward
        done = self.done
        return current_pos, next_pos, skills, obs, next_obs, actions, rewards, done

    def wipe(self):
        super(SkillDynamicsMemory, self).__init__(memory_length=self.memory_length, device=self.device)
        self.skills = torch.tensor([], dtype=torch.int, device=self.device, requires_grad=False)
        self.current_pos = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=False)
        self.next_pos = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=False)

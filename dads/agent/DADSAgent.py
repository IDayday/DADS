import torch
import torch.nn.functional as funcs
from random import randint
from soft_actor_critic.agent.SACAgent import SACAgent
from dads.agent.SkillDynamics import SkillDynamics
from dads.agent.SkillDynamicsMemory import SkillDynamicsMemory
from datetime import datetime
import logging
import time

class DADSAgent(SACAgent):
    def __init__(self, env, device, n_skills, pos_shape, train=False, log_dir=None, learning_rate=3e-4, memory_length=1e5, batch_size=32, num_hidden_neurons=256):
        self.n_skills = n_skills
        self.active_skill = None
        env_state_length = env.observation_space.shape[0]
        input_shape_with_skill_encoder = env_state_length + n_skills
        super(DADSAgent, self).__init__(env=env, device=device, input_shape=input_shape_with_skill_encoder,
                                        num_hidden_neurons=num_hidden_neurons, writer=None, learning_rate=learning_rate,
                                        discount_rate=0.99, memory_length=memory_length, batch_size=batch_size,
                                        polyak=0.995, alpha=0.1, policy_train_delay_modulus=2)
        self.skill_dynamics = SkillDynamics(env_shape=env_state_length, skill_shape=n_skills, output_shape=env_state_length,
                                            device=self.device, pos_shape=pos_shape, num_hidden_neurons=num_hidden_neurons,
                                            learning_rate=3e-4)
        self.batch_size = batch_size
        self.total_games = 0
        # Overwrite the memory that the SACAgent instantiates with a new memory that additionally stores the skills:
        self.memory = SkillDynamicsMemory(memory_length=self.memory_length, device=self.device)
        if train:
            self.logging = logging.basicConfig(filename=log_dir, filemode='a', level=logging.INFO)

    def save_models(self, save_path):
        torch.save({"skill_dynamics": self.skill_dynamics.state_dict(),
                    "actor": self.actor.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic2": self.critic2.state_dict(),
                    "critic_target1": self.critic_target1.state_dict(),
                    "critic_target2": self.critic_target2.state_dict()
                    }, save_path + "/params.pth" )


    def load_models(self, load_path):
        checkpoint = torch.load(load_path)
        self.skill_dynamics.load_state_dict(checkpoint["skill_dynamics"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic_target1.load_state_dict(checkpoint["critic_target1"])
        self.critic_target2.load_state_dict(checkpoint["critic_target2"])


    def _sample_skill(self, skill):
        if skill is None:
            skill = randint(0, self.n_skills - 1)
        skill_one_hot_encoded = self._create_skill_encoded([skill])
        self.active_skill = skill_one_hot_encoded
        self.skill_num = skill

    def _create_skill_encoded(self, skill_ints):
        skill_one_hot_encoder = torch.zeros((len(skill_ints), self.n_skills), dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(skill_ints)):
            skill_one_hot_encoder[i][skill_ints[i]] = 1.0
        return skill_one_hot_encoder

    def play_games(self, num_games, verbose=False, display_gameplay=False, skill=None, train=True):
        for _ in range(num_games):
            self._play_game(verbose, display_gameplay, skill, train)

    def _play_game(self, verbose, display_gameplay, skill=None, train=True):
        self.total_games += 1
        print("total games: ", self.total_games)
        self.env.reset_env()
        self.memory.wipe()      # Start each episode with an empty memory
        self._sample_skill(skill)  # Updates self.active_skill tensor to be a newly sampled one hot encoding
        memorized_batch_size = 0
        env_timesteps = 0
        episode_reward = 0
        start = time.time()
        while not self.env.done or (env_timesteps < 250 and train):
            env_timesteps += 1
            # Take the observation from the environment, format it, push it to GPU
            current_obs = torch.tensor(self.env.observation, dtype=torch.float, device=self.device, requires_grad=False).reshape((1, -1))
            # current_pos = torch.tensor(self.env.xy_coords, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, -1)
            current_pos = torch.tensor(self.env.env.sim.data.body_xpos, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, -1)
            current_obs_skill = torch.cat((current_obs, self.active_skill), 1)
            current_action = self.choose_action(current_obs_skill)
            if display_gameplay:
                self.env.env.render()
            self.env.take_action(current_action.squeeze().cpu().numpy())
            next_obs = torch.tensor(self.env.observation, dtype=torch.float, device=self.device, requires_grad=False).reshape((1, -1))
            done = torch.tensor([[self.env.done]], dtype=torch.int, device=self.device, requires_grad=False)
            reward = torch.tensor([[self.env.reward]], dtype=torch.int, device=self.device, requires_grad=False)
            # next_pos = torch.tensor(self.env.xy_coords, dtype=torch.float, device=self.device, requires_grad=False).reshape(1,-1)
            next_pos = torch.tensor(self.env.env.sim.data.body_xpos, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, -1)
            episode_reward += self.env.reward
            self.memory.append("current_pos", current_pos)
            self.memory.append("next_pos", next_pos)
            self.memory.append("skills", self.active_skill)
            self.memory.append("observation", current_obs)
            self.memory.append("next_observation", next_obs)
            self.memory.append("action", current_action)
            self.memory.append("reward", reward)
            self.memory.append("done", done)
            memorized_batch_size += 1
            if memorized_batch_size >= self.batch_size and train:
                self.train_models(verbose=verbose)
                self.memory.wipe() # On-Policy?
                memorized_batch_size = 0
        episode_time = time.time() - start
        info = 'Episode {} Finished | Active Skill: {:d} |Reward: {:.2f} | Step: {:d} | Costtime: {:.2f}'.format(
                    self.total_games, self.skill_num, episode_reward, env_timesteps, episode_time)
        logging.info(info)
        

    def train_models(self, verbose):
        if len(self.memory.observation) < self.batch_size:
            return
        self.updates += 1
        current_pos, next_pos, skills, states, new_states, actions, _, dones = self.memory.sample_memory()
        pos_shape = current_pos.shape[1]
        states_and_skills = torch.cat((states, skills), 1)
        new_states_and_skills = torch.cat((states, skills), 1)
        pos_states_and_skills = torch.cat((current_pos, states, skills), 1)
        new_pos_states_and_skills = torch.cat((next_pos, new_states, skills), 1)
        for _ in range(32):
            self.skill_dynamics.train_model(pos_states_and_skills, new_pos_states_and_skills, verbose=verbose)

        # For DADS, we calculate an intrinsic reward rather than using the rewards sampled from the memory (and blanked
        # out above):
        rewards = self.calc_intrinsic_rewards(skills=skills, states=states, new_states=new_states,
                                              current_pos=current_pos, next_pos=next_pos)
        for _ in range(128):
            next_actions, next_log_probs = self.actor.sample_normal(observation=new_states_and_skills, reparameterise=True)
            target_q1 = self.critic_target1.forward(observation=new_states_and_skills, action=next_actions)
            target_q2 = self.critic_target2.forward(observation=new_states_and_skills, action=next_actions)
            next_min_q = torch.min(target_q1, target_q2) - self.alpha.to(self.device) * next_log_probs
            next_q = rewards.view(-1, 1) + self.discount_rate * (1 - dones) * next_min_q

            curr_q1 = self.critic1.forward(observation=states_and_skills, action=actions)
            curr_q2 = self.critic2.forward(observation=states_and_skills, action=actions)
            q1_loss = funcs.mse_loss(curr_q1, next_q.detach())
            q2_loss = funcs.mse_loss(curr_q2, next_q.detach())

            self.critic1.optimizer.zero_grad(set_to_none=True)
            q1_loss.backward()
            self.critic1.optimizer.step()

            self.critic2.optimizer.zero_grad(set_to_none=True)
            q2_loss.backward()
            self.critic2.optimizer.step()
            if verbose:
                print("critic1 loss:", q1_loss, "critic2 loss:", q2_loss)

            curr_actions, curr_log_probs = self.actor.sample_normal(observation=states_and_skills, reparameterise=True)
            if self.updates % self.policy_train_delay_modulus == 0:
                curr_q1_policy = self.critic1.forward(observation=states_and_skills, action=curr_actions)
                curr_q2_policy = self.critic2.forward(observation=states_and_skills, action=curr_actions)
                curr_min_q_policy = torch.min(curr_q1_policy, curr_q2_policy)
                policy_loss = torch.mean(self.alpha * curr_log_probs - curr_min_q_policy)

                self.actor.optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                self.actor.optimizer.step()
                if verbose:
                    print("actor loss:", policy_loss)

                self.polyak_soft_update(target=self.critic_target1, source=self.critic1)
                self.polyak_soft_update(target=self.critic_target2, source=self.critic2)
                del policy_loss, curr_min_q_policy, curr_q2_policy, curr_q1_policy


    def calc_intrinsic_rewards(self, skills, states, new_states, current_pos, next_pos):
        batch_length = states.shape[0]
        with torch.no_grad():
            # Need to get the probability of the current state given the previous state and skill
            states_and_skills = torch.cat((states, skills), 1)
            _, _, _, this_skill_distribution = self.skill_dynamics.sample_next_state(states_and_skills)
            current_pos_states = torch.cat((current_pos, states, ), 1)
            next_pos_states = torch.cat((next_pos, new_states), 1)
            this_skill_log_probs = this_skill_distribution.log_prob(next_pos_states - current_pos_states)

            # We need the exponent of the log probs here
            summed_other_skill_probs = torch.zeros(size=(batch_length, 1), dtype=torch.float, device=self.device)
            for i in range(self.n_skills):
                other_skill = self._create_skill_encoded([i])
                states_and_other_skills = torch.cat((states, other_skill.repeat((batch_length, 1))), 1)
                _, _, _, other_skill_distribution = self.skill_dynamics.sample_next_state(states_and_other_skills)
                other_skill_probs = torch.exp(other_skill_distribution.log_prob(next_pos_states - current_pos_states))
                summed_other_skill_probs += other_skill_probs.reshape(-1, 1)

            # our original intrinsic_reward constructed to match the paper:
            # intrinsic_reward = this_skill_log_probs.view(-1, 1) - summed_other_skill_log_probs +\
            #                    torch.log(torch.tensor([4.], dtype=torch.float, device=self.device))

            # intrinsic_reward from their TensorFlow implementation of the paper on GitHub:
            # intrinsic_reward = np.log(num_reps + 1) -\
            # np.log(1 + np.exp(np.clip(logp_altz - logp.reshape(1, -1), -50, 50)).sum(axis=0))

            # A mix between our approach and their implementation, so deviates from the paper somewhat
            # intrinsic_reward = torch.log(torch.tensor([4.], dtype=torch.float, device=self.device)) +\
            #                     torch.clamp(this_skill_log_probs.view(-1, 1) - torch.log(summed_other_skill_probs), -50, 50)

            # Copying their implementation of the rewards but in PyTorch:
            intrinsic_reward = torch.log(torch.tensor([4.], dtype=torch.float, device=self.device)) -\
            torch.log(1 + torch.exp(torch.clamp(torch.log(summed_other_skill_probs) - this_skill_log_probs.view(-1, 1), -50, 50)))
            return intrinsic_reward



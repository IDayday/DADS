from turtle import pos
import torch
import torch.nn.functional as funcs
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch import nn
from torch import optim


class SkillDynamics(nn.Module):
    def __init__(self, env_shape, skill_shape, output_shape, device, pos_shape, num_hidden_neurons=256, learning_rate=3e-4):
        super(SkillDynamics, self).__init__()
        self.env_shape = env_shape
        self.skill_shape = skill_shape
        self.input_shape = env_shape + skill_shape # This includes the skill encoder
        self.pos_shape = pos_shape
        self.output_shape = output_shape + pos_shape  # NOTE: we predict the delta x and delta y, despite not inputting (x ,y)! adv: I change (x,y) to pos
        self.device = device
        self.num_hidden_neurons = num_hidden_neurons  # Paper uses same number across layers and networks
        self.learning_rate = learning_rate
        # The first layer takes the state input plus a one-hot encoding of which skill is active
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.input_shape)
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.num_hidden_neurons)
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.num_hidden_neurons)
        self.fc1 = nn.Linear(self.input_shape, self.num_hidden_neurons)
        self.fc2 = nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
        # Each expert is a multinomial gaussian, with the same dimension as the input state space
        self.expert1_mu = nn.Linear(self.num_hidden_neurons, self.output_shape)
        self.expert2_mu = nn.Linear(self.num_hidden_neurons, self.output_shape)
        self.expert3_mu = nn.Linear(self.num_hidden_neurons, self.output_shape)
        self.expert4_mu = nn.Linear(self.num_hidden_neurons, self.output_shape)
        # A softmax layer is used as part of gating model to decide when to use which expert. This is currently
        # implemented a Linear layer, but we should confirm this is the same as in the paper.
        # The softmax acts to choose which expert to use, there's 4 hardcoded experts, so 4 output neurons:
        self.softmax_input = nn.Linear(self.num_hidden_neurons, skill_shape)
        # # All experts use an identity matrix as the standard deviation.
        self.sigma = torch.eye(self.output_shape, requires_grad=False, device=self.device)
        # We don't want to update this so no gradients
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, observation_and_skill):
        out = self.batchnorm0(observation_and_skill)
        out = self.fc1(out)
        out = self.batchnorm1(out)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = funcs.relu(out)
        mu1 = self.expert1_mu(out)
        mu2 = self.expert2_mu(out)
        mu3 = self.expert3_mu(out)
        mu4 = self.expert4_mu(out)
        softmax_gate = funcs.softmax(self.softmax_input(out), dim=1)
        return mu1, mu2, mu3, mu4, softmax_gate

    def sample_next_state(self, state_and_skill, reparam=False):
        mu1, mu2, mu3, mu4, softmax_gate = self.forward(state_and_skill)
        _, top_gate_ind = torch.topk(softmax_gate, 1)

        # Here we take ith expert's prediction of the mean multiplied by the softmax value IF the ith softmax was
        # the maximum. The other mus are "added" but are zero because they are multiplied by False.
        expert_mu = (mu1.T * softmax_gate[:, 0]).T * (top_gate_ind == 0) + \
                       (mu2.T * softmax_gate[:, 1]).T * (top_gate_ind == 1) + \
                       (mu3.T * softmax_gate[:, 2]).T * (top_gate_ind == 2) + \
                       (mu4.T * softmax_gate[:, 3]).T * (top_gate_ind == 3)

        # rsample() uses reparameterisation trick, so that the gradients can flow backwards through the parameters
        # in contrast to sample() which blocks gradients (as it's a random sample)
        # TODO: MultivariateNormal appears to have a memory leak, replacing as we're only using an identity matrix as
        #  the variance anyway. This appears on an AMD Vega 56 and doesn't replicate on Google Colab.
        # next_state_distribution = MultivariateNormal(expert_mu, self.sigma)
        next_state_distribution = Independent(Normal(expert_mu, self.sigma[0, 0]), 1)
        if reparam:
            delta = next_state_distribution.rsample()
        else:
            delta = next_state_distribution.sample()

        delta_pos = delta[:, 0:self.pos_shape]
        delta_state = delta[:, self.pos_shape:]

        next_state = state_and_skill[:, 0:self.env_shape] + delta_state
        return next_state, delta, delta_pos, next_state_distribution

    def train_model(self, current_pos_state_and_skill, next_pos_state_and_skill, verbose=False):
        self.optimizer.zero_grad(set_to_none=True)
        current_state_and_skill = current_pos_state_and_skill[:, self.pos_shape:]
        # Key nuance here: The output includes delta x and delta y, but we don't input x and y.
        pred_next_state, delta, delta_pos, distribution = self.sample_next_state(current_state_and_skill, reparam=True)
        # We index after this difference as we only want delta of the state and not the space-skill concatenation
        # We want to predict the delta of x and y, so they stay in here, but we remove the skill from the end of the matrix
        actual_delta = (next_pos_state_and_skill - current_pos_state_and_skill)[:, :-self.skill_shape]
        # columnwise_stds = torch.std(actual_delta, dim=0)
        # columnwise_stds[columnwise_stds == 0] = 1
        # actual_delta_scaled = ((actual_delta - torch.mean(actual_delta, dim=0)) / columnwise_stds)
        loss = -1. * torch.mean(distribution.log_prob(actual_delta))
        loss.backward()
        if verbose:
            print("Skill dynamics loss: ", loss.item())
        self.optimizer.step()

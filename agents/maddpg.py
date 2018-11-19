import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents.utils import soft_update


class MADDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, memory, network, device,
                GRADIENT_CLIP = 1,
                ACTIVATION = F.relu,
                BOOTSTRAP_SIZE = 5,
                GAMMA = 0.99, 
                TAU = 1e-3, 
                LR_CRITIC = 5e-4,
                LR_ACTOR = 5e-4, 
                UPDATE_EVERY = 1,
                TRANSFER_EVERY = 2,
                UPDATE_LOOP = 10,
                ADD_NOISE_EVERY = 5,
                WEIGHT_DECAY = 0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents: number of running agents
            memory: instance of ReplayBuffer
            network: a class inheriting from torch.nn.Module that define the structure of the actor critic neural network
            device: cpu or cuda:0 if available
            ACTIVATION: the activation function to be used by the network
            BOOTSTRAP_SIZE: length of the bootstrap
            GAMMA: discount factor
            TAU: for soft update of target parameters
            LR_CRITIC: learning rate of the critic network
            LR_ACTOR: learning rate of the actor network
            UPDATE_EVERY: how often to update the networks
            TRANSFER_EVERY: after how many update do we transfer from the online network to the targeted fixed network
            UPDATE_LOOP: number of update loop whenever the networks are being updated
            ADD_NOISE_EVERY: how often to add noise to favor exploration
            WEIGHT_DECAY: parameter of the Adam Optimizer of the critic network
            GRADIENT_CLIP: limit of exploding gradient to be clipped
        """
        
        # Actor networks
        
        self.network_local = network(state_size, action_size, state_size * 2 , activation = ACTIVATION).to(device)
        self.network_target = network(state_size, action_size, state_size * 2 , activation = ACTIVATION).to(device)
        self.actor_optim = optim.Adam(self.network_local.actor.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_optim = optim.Adam(self.network_local.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Ensure that at the begining, both target and local are having the same parameters
        soft_update(self.network_local, self.network_target, 1)
        
        self.device = device
        
        # Noise
        self.noise = None
        
        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.n_step = 0
        
        self.BOOTSTRAP_SIZE = BOOTSTRAP_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.UPDATE_LOOP = UPDATE_LOOP
        self.ADD_NOISE_EVERY = ADD_NOISE_EVERY
        self.GRADIENT_CLIP = GRADIENT_CLIP
        
        # initialize these variables to store the information of the n-previous timestep that are necessary to apply the bootstrap_size
        
        self.rewards = deque(maxlen=BOOTSTRAP_SIZE)
        self.states = deque(maxlen=BOOTSTRAP_SIZE)
        self.actions = deque(maxlen=BOOTSTRAP_SIZE)
        self.gammas = np.array([[GAMMA ** i for j in range(num_agents)] for i in range(BOOTSTRAP_SIZE)])
        self.loss_function = torch.nn.SmoothL1Loss()
    
    def reset(self):
        if self.noise:
            for n in self.noise:
                n.reset()
        
    def set_noise(self, noise):
        self.noise = noise
        
    def save(self, filename):
        torch.save(self.network_local.state_dict(),"weights/{}_local.pth".format(filename))
        torch.save(self.network_target.state_dict(),"weights/{}_target.pth".format(filename))
     
    def load(self, path):
        self.network_local.load_state_dict(torch.load(path + "_local.pth"))
        self.network_target.load_state_dict(torch.load(path + "_target.pth"))
    
    def act(self, states, noise = 0.0):
        """Returns actions of each actor for given states.
        
        Params
        ======
            state (array_like): current states
            add_noise: either alter the decision of the actor network or not. During training, this is necessary to promote the exploration. However, during validation, this is altering the agent and should be deactivated.
        """
        ret = None
        
        self.n_step = (self.n_step + 1) % self.ADD_NOISE_EVERY
        
        with torch.no_grad():
            self.network_local.eval()
            states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
            ret = self.network_local(states).squeeze().cpu().data.numpy()
            self.network_local.train()
            if self.n_step == 0:
                for i in range(len(ret)):
                    ret[i] += noise * self.noise[i].sample()
        return ret
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        
        self.rewards.append(rewards)
        self.states.append(states)
        self.actions.append(actions)
            
        if len(self.rewards) == self.BOOTSTRAP_SIZE:
            # get the sum of rewards per agents
            reward = np.sum(self.rewards * self.gammas, axis = -2)
            self.memory.add(self.states[0], self.actions[0], reward, next_states, dones)
            
        if np.any(dones):
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            
        # Learn every UPDATE_EVERY timesteps
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY
        
        t_step=0
        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            for _ in range(self.UPDATE_LOOP):
                self.learn()
                t_step=(t_step + 1) % self.TRANSFER_EVERY
                if t_step == 0:
                    soft_update(self.network_local, self.network_target, self.TAU)
    
    def build_full_state(self, states):
        state_size = states.shape[-1]
        num_agents = states.shape[-2]
        full_states = torch.zeros((states.shape[0], num_agents, state_size * num_agents))
        for i in range(num_agents):
            start = 0
            idx = np.arange(num_agents)
            for j in idx:
                full_states[:,i,start:start + state_size] += states[:,j]
                start += state_size
            idx = np.roll(idx, -1)
        return full_states
    
    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        # sample the memory to disrupt the internal correlation
        states, actions, rewards, next_states, dones = self.memory.sample()
        full_states = self.build_full_state(states)

        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        with torch.no_grad():
            self.network_target.eval()
            next_actions = self.network_target(next_states)
            next_full_states = self.build_full_state(next_states)
            # the rewards here was pulled from the memory. Before being registered there, the rewards are already considering the size of the bootstrap with the appropriate discount factor
            q_next = self.network_target.estimate(next_full_states, next_actions).squeeze(-1)
            assert q_next.shape == dones.shape, " q_next {} != dones {}".format(q_next.shape, dones.shape)
            assert q_next.shape == rewards.shape, " q_next {} != rewards {}".format(q_next.shape, rewards.shape)
            targeted_value = rewards + (self.GAMMA**self.BOOTSTRAP_SIZE)*q_next*(1 - dones)
         
        current_value = self.network_local.estimate(full_states, actions).squeeze(-1)
        assert targeted_value.shape == current_value.shape, " targeted_value {} != current_value {}".format(targeted_value.shape, current_value.shape)
        
        # calculate the loss of the critic network and backpropagate
        self.critic_optim.zero_grad()
        loss = self.loss_function(current_value, targeted_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network_local.critic.parameters(), self.GRADIENT_CLIP)
        self.critic_optim.step()

        # optimize the actor by having the critic evaluating the value of the actor's decision
        self.actor_optim.zero_grad()
        actions_pred = self.network_local(states)
        mean = self.network_local.estimate(full_states, actions_pred).mean()
        # during the back propagation, parameters of the actor that led to a bad note from the critic will be demoted, and good parameters that led to a good note will be promoted
        (-mean).backward()
        self.actor_optim.step()    
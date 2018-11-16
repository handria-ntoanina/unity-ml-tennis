import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import time

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    θ_target = θ_target + τ*(θ_local - θ_target)
    θ_local = r + gamma * θ_local(s+1)

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    # this is transferring gradually the parameters of the online Q Network to the fixed one
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ActionNoise:
    """ Noise generator that disturb the output of a network """
    def __init__(self, size, device, seed, mu=0., theta=0.15, sigma=0.2):
        self.noise = OUNoise(size, device, seed, mu, theta, sigma)
        
    def reset(self):
        self.noise.reset()
        
    def apply(self, model, state):
        action = model(state).cpu().data.numpy()
        action += self.noise.sample()
        return np.clip(action, -1, 1)
    
class ParameterNoise:
    """ Noise generator that disturb the weight of a network and thus its output """
    def __init__(self, model, device, seed, mu=0., theta=0.15, sigma=0.2):
        size = sum([np.array(param.data.size()).prod() for param in model.parameters()])
        self.noise = OUNoise(size, device, seed)
        
    def reset(self):
        self.noise.reset()
        
    def apply(self, model, state):
        noise_sample = self.noise.sample()
        # apply a noise to the parameters only for exploration purpose
        self.apply_noise(model, noise_sample)
        ret = model(state).cpu().data.numpy()
        # restore the previous parameters otherwise the noise will disturb the acquired knowldege
        self.apply_noise(model, -noise_sample)
        return ret
    
    def apply_noise(self, model, noise_sample):
        start = torch.tensor(0)
        for param in model.parameters():
            size = np.array(param.data.size()).prod().item()
            noise = noise_sample[start:start + size].reshape(param.data.size()).float()
            param.data.add_(noise)
            start += size
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, device, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.device = device
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return torch.from_numpy(self.state).to(self.device).requires_grad_(False)
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, device, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
       
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return self.to_tensor(experiences)
    
    def shuffle_all(self):
        temp = list(self.memory)
        random.shuffle(temp)
        batch_count = int(len(temp) / self.batch_size)
        for a in range(batch_count):
            yield self.to_tensor(temp[a:(a+1)*self.batch_size])
    
    def to_tensor(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device).requires_grad_(False)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device).requires_grad_(False)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import gym
import collections
import torch
import torch.nn.functional as F


class replay_uffer():
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
    
    def add_element(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions = random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.layer1 = torch.nn.Linear(state_dim,hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self,x):
        return self.layer2(F.relu(self.layer1(x)))
    
    
class DQN():  
    def __init__(self,env,state_dim,hidden_dim,action_dim,learning_rate,gamma,eplison,target_update,device):
        self.env = env
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eplison = eplison
        self.target_update = target_update
        self.device = device
        
        self.q_net = Qnet(state_dim,hidden_dim,action_dim).to(device)      #qnet
        self.target_net = Qnet(state_dim,hidden_dim,action_dim).to(device) #target net
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr = self.learning_rate)
        self.update_count = 0
        
    def take_action(self,state):
        if random.uniform(0,1) < self.eplison:
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        else:
            action = np.random.randint(self.action_dim)
        
        return action
    
    def update(self,transition_dict):
        states = torch.tensor(transition_dict["state"],dtype=torch.float).to(
            self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        q_values = self.q_net(states).gather(1,actions)
        
        max_next_q_values = self.target_net(next_states).max(1)[0].view(-1,1)
    
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad() 
        dqn_loss.backward()
        
        self.optimizer.step()
        
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1        
        

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.99
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
iteration_number = 10
        
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    
    print("----------------------------------------------------")
    print(f"lr = {lr}, num_episodes = {num_episodes}, hidden_dim = {hidden_dim}, discount factor = {gamma},batch size = {batch_size}")
    print(f"epsilon is {epsilon}, target_update = {target_update}, buffer_size = {buffer_size}, minimal_size = {minimal_size}")
    if(torch.cuda.is_available):
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("----------------------------------------------------")
    
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    buffer = replay_uffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(env,state_dim,hidden_dim,action_dim,gamma,epsilon,target_update,device)
    return_list = []
    
    print("start training")
    for i in range(iteration_number):
        print(f"----->iteraion {i}")
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for episode in range(int(num_episodes / 10)):
                total_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,_ = env.step(action)
                    buffer.add_element(state,action,reward,next_state,done)
                    state = next_state
                    total_return += reward
                    
                    if buffer.size() > minimal_size: #then start trainging
                        b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update()
                return_list.append(total_return)
    
    
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on CartPole-v1')
    plt.show()
                    
        
    
    
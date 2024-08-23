import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import gym


class sarsa():
    def __init__(self,env,observation_space_size,action_space_size,max_step,episodes_num,gamma,alpha,epsilon):
        self.env = env
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.max_step = max_step
        self.episodes_num = episodes_num
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.q_map = np.zeros((self.observation_space_size,self.action_space_size))
        self.mean_return = np.zeros(0)
    
    
    def take_action(self,state):
        if random.uniform(0,1) < self.epsilon:
            state_all = self.q_map[state,:]
            max_value = np.max(state_all)
            max_index = np.where(state_all == max_value)[0]
            action = np.random.choice(max_index)
        else:
            action = self.env.action_space.sample()
        
        return action
            
    def get_action_name(self,action):
        if action == 0:
            return "left"
        elif action == 1:
            return "down"
        elif action == 2:
            return "right"
        else:
            return "up"
    
    def genarate(self):
        trace = []
        for episode in range(self.episodes_num):

            trace.clear()
            state = self.env.reset()
            state = state[0]
            step = 0
            is_done = False
            total_return = 0
            action = self.take_action(state)
            
            for step in range(self.max_step):
                
                trace.append(self.get_action_name(action))
                next_state,reward,is_done,truncated,info = self.env.step(action)
                
                next_action = self.take_action(next_state)
                
                self.q_map[state][action] = self.q_map[state][action] + self.alpha*(reward + self.gamma * self.q_map[next_state,next_action] - self.q_map[state][action])
                total_return += self.q_map[state][action]
                if state == next_state:
                   self.q_map[state, action] = 0
                    
                if is_done == True:
                    if reward == 1:
                        self.q_map[state,action] = 1
                    else:
                        self.q_map[state,action] = 0
                    self.mean_return = np.append(self.mean_return,total_return/step)
                    break
                    
                state = next_state
                action = next_action
            
            
                
            if self.epsilon <= 1:
                self.epsilon += 0.05
        
        print(f"the last trace is :{trace}")
        #print(self.q_map)
        return self.mean_return
                


episodes_num = 6000
max_step = 400
iteration_num = 10
learn_rate = 0.8
discount_rate = 0.99
epsilon = 0.3

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    print(env.observation_space)
    observation_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    
    print(f"size of observation space: {observation_space_size}, size of action space: {action_space_size}")
    
    q_learning_instance = sarsa(env,observation_space_size,action_space_size,max_step,episodes_num,discount_rate,learn_rate,epsilon)
    mean_return = q_learning_instance.genarate()

    variance = np.var(mean_return)
   
    plt.figtext(0.5, 0.95,f"the variance = {variance}", ha='center')
    plt.title("Sarsa")
    plt.xlabel("episode")
    plt.ylabel("mean return")
    plt.plot(mean_return)    
    plt.show()
    

        
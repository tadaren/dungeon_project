import numpy as np
import random
import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from collections import namedtuple
Transition = namedtuple('Transicion', ('state', 'action', 'next_state', 'reward'))

from simulator import AdvancedSimulator3
from learning import RoomSelector

GAMMA = 0.99
MAX_STEP = 1500
NUM_EPISODES = 80000
PRINT_EVERY_EPISODE = 5000

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
    
    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 32
CAPACITY = 10000

class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, num_actions)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        output = self.fc5(h4)
        return output

class Net2(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(num_states, num_states)
        self.fc2 = nn.Linear(num_states, num_states)
        self.fc3 = nn.Linear(num_states, num_states)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        # self.model = Net(num_states, num_actions)
        self.model = Net2(num_states, num_actions)

        self.target_net = copy.deepcopy(self.model)
        self.target_net.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # print(batch.reward)
        # print(batch.action)
        # print(type(batch.reward))
        # print(type(batch.action))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        rewatd_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)

        next_state_values = torch.zeros(BATCH_SIZE)

        self.target_net.eval()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + rewatd_batch

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def decide_action(self, state, episode):
        epsilon = 0.41 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]
            )
        return action

    def brain_predict(self, state):
        self.model.eval()
        with torch.no_grad():
            action = self.model(state).max(1)[1].view(1, 1)
            return action

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_network(self):
        self.brain.replay()

    def update_target_model(self):
        self.brain.update_target_model()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)
    
    def predict_action(self, state):
        action = self.brain.brain_predict(state)
        return action

class Environment:
    def __init__(self):
        self.simulator = AdvancedSimulator3()
        self.room_state_size = ((self.simulator.dungeon.max_room_size[0]+2), (self.simulator.dungeon.max_room_size[1]+2))
        # self.num_states = self.room_state_size[0]*self.room_state_size[1]
        self.num_states = 2+2+2+1
        self.num_actions = 5
        self.agent = Agent(self.num_states, self.num_actions)
        self.room_selector = None

    def info2state(self, info):
        state = np.zeros(self.room_state_size)
        room_id = info['roomId']
        room_origin = info['map']['rooms'][room_id]['origin']
        room_size = info['map']['rooms'][room_id]['size']
        room_road_ends = info['map']['rooms'][room_id]['roadEnds']
        
        state[1:room_size[0]+1, 1:room_size[1]+1] = 1
        x, y = info['x'], info['y']
        state[y-room_origin[0]+1, x-room_origin[1]+1] = 2
        enemies = info['enemies']
        for enemy in enemies:
            e_x, e_y = enemy['x'], enemy['y']
            if e_x == -1 or e_y == -1:
                continue
            state[e_y-room_origin[0]+1, e_x-room_origin[1]+1] = 3
        
        index = -1
        if room_id == info['goalRoomId']:
            goal_x, goal_y = info['goalPosition']
            state[goal_y-room_origin[0]+1, goal_x-room_origin[1]+1] = 4
        else:
            x, y, index = self.room_selector.next(info)
            end = room_road_ends[index]
            state[end[1]-room_origin[0]+1, end[0]-room_origin[1]+1] = 4
        state = torch.from_numpy(state.reshape(-1)).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)
        return state, index
    
    def info2state2(self, info):
        room_id = info['roomId']
        x, y = info['x'], info['y']
        d_x, d_y, index = self.room_selector.next(info)
        state = [d_x - x + 20, d_y - y + 15]
        for enemy in info['enemies']:
            e_x, e_y = enemy['x'], enemy['y']
            if e_x == -1 or e_y == -1:
                state.append(20)
                state.append(15)
                continue
            state.append(e_x - x + 20)
            state.append(e_y - y + 15)
        state.append(info['map']['rooms'][room_id]['roads'][index]['align'])
        
        state = torch.FloatTensor(state)
        state = torch.unsqueeze(state, 0)
        return state, index


    def run(self):
        for episode in range(NUM_EPISODES):
            info = self.simulator.info()
            self.room_selector = RoomSelector(info)
            
            # state, index = self.info2state(info)
            state, index = self.info2state2(info)

            if episode % 15 == 0:
                self.agent.update_target_model()

            turn = 0
            reward_sum = 0

            while not info['isEnd']:
                turn += 1

                action = self.agent.get_action(state, episode)
                # print(int(action))
                reward = self.simulator.action({
                    'action': int(action),
                    'roadId': index
                })

                next_info = self.simulator.info()
                # next_state, index = self.info2state(next_info)
                next_state, index = self.info2state2(next_info)

                if next_info['isEnd']:
                    next_state = None
                
                reward_sum += reward

                self.agent.memorize(state, action, next_state, torch.FloatTensor([[reward]]))
                self.agent.update_q_network()

                state = next_state
                info = next_info

            
            if episode % PRINT_EVERY_EPISODE == 0:
                print(f'{episode} / {NUM_EPISODES} reward: {reward_sum} turn: {turn}')
                self.simulator.save()

            self.simulator.reset()

    def save_model(self):
        torch.save(self.agent.brain.model.state_dict(), 'weight.pth')


if __name__ == "__main__":
    env = Environment()
    env.run()
    env.save_model()
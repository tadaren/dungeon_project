import sys
import time

from simulator import AdvancedSimulator3
from learning import RoomSelector, state_tuple, select_action
import numpy as np

simulator = AdvancedSimulator3()


def play(q, loop_num):
    for step in range(loop_num):
        state = simulator.info()
        room_selector = RoomSelector(state)
        s, index = state_tuple(state, room_selector)
        sum_reward = 0
        turn = 0

        while not state['isEnd']:
            turn += 1
            action = select_action(
                q[s],
                0
            )
            action = int(action)
            reward = simulator.action({'action': action, 'roadId': index})
            sum_reward += reward

            next_state = simulator.info()
            next_s, index = state_tuple(next_state, room_selector)
            state = next_state
            s = next_s

        simulator.reset()
        time.sleep(1)


if __name__ == '__main__':
    q_table = np.load(sys.argv[1])
    N = int(sys.argv[2])
    play(q_table, N)

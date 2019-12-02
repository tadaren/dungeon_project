import random

from simulator import AdvancedSimulator2, AdvancedSimulator3
import numpy as np


class RoomSelector:
    def __init__(self, state):
        self.road_visit_count = {road: 0 for road in state['roadUsed']}
        self.before_room_id = state['roomId']

    def next(self, state):
        room_id = state['roomId']
        room = state['map']['rooms'][room_id]
        if room_id != self.before_room_id:
            for road in room['roads']:
                if road['room1Id'] == room_id and road['room2Id'] == self.before_room_id or road['room1Id'] == \
                        self.before_room_id and road['room2Id'] == room_id:
                    self.road_visit_count[road['id']] += 1
            self.before_room_id = room_id
        if room_id == state['goalRoomId']:
            x, y = state['goalPosition']
            return x, y, -1
        roads = [(self.road_visit_count[road['id']], i) for i, road in enumerate(room['roads'])]
        _, index = sorted(roads)[0]
        x, y = room['roadEnds'][index]
        return x, y, index


def select_action(actions: np.ndarray, eps):
    r = random.random()
    if r < eps:
        return random.randint(0, len(actions)-1)
    return actions.argmax()


def enemy_state(state):
    result = []
    for e in state['enemies']:
        if e['x'] != -1 and e['y'] != -1:
            x = e['x'] - state['x']
            y = e['y'] - state['y']
            if -2 <= x <= 2 and -2 <= y <= 2:
                result.append((
                    x + 2,
                    y + 2
                ))
    while len(result) < 2:
        result.append((2, 2))
    return result


def state_tuple(state, selector):
    enemies = enemy_state(state)
    x = state['x']
    y = state['y']
    destination_x, destination_y, index = selector.next(state)
    if index == -1:
        align = 0
    else:
        align = state['map']['rooms'][state['roomId']]['roads'][index]['align']
    return (
        destination_x - x + 20,
        destination_y - y + 15,
        align,
        enemies[0][0],
        enemies[0][1],
        enemies[1][0],
        enemies[1][1]
    ), index


def main():
    random.seed(14)
    simulator = AdvancedSimulator3()
    alpha = 0.1
    gamma = 0.8
    q = np.random.random((43, 33, 2, 5, 5, 5, 5, 5))
    eps = np.full((43, 33, 2, 5, 5, 5, 5), 0.99)

    max_step = 1000000
    for step in range(max_step):
        state = simulator.info()
        room_selector = RoomSelector(state)
        s, index = state_tuple(state, room_selector)
        sum_reward = 0
        turn = 0

        while not state['isEnd']:
            turn += 1
            action = select_action(
                q[s],
                eps[s]
            )
            action = int(action)
            eps[s] *= 0.999
            reward = simulator.action({'action': action, 'roadId': index})
            sum_reward += reward

            next_state = simulator.info()
            next_s, index = state_tuple(next_state, room_selector)
            ss = (*s, action)

            q[ss] = (1.0-alpha)*q[ss] + alpha*(reward + gamma*q[next_s].max())
            state = next_state
            s = next_s

        simulator.reset()
        print(step, '/', max_step, 'reward:', sum_reward, 'turn:', turn)
        # simulator.dungeon.print_floor_map()
        # time.sleep(1)


if __name__ == '__main__':
    main()

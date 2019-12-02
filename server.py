import random
from typing import Dict
from bottle import *
from simulator import Simulator, RoomGraphSimulator, CellMoveSimulator, Simulator2, AdvancedSimulator2

simulators: Dict[int, Simulator] = {}


@get('/')
def index():
    return static_file('index.html', './static')


@get('/<filename>')
def file(filename):
    return static_file(filename, './static')


@post('/init')
def init():
    data = request.json
    seed = data.get('seed', None)
    mode = data['mode']
    if seed is not None:
        random.seed(seed)

    if mode == 1:
        simulator = RoomGraphSimulator()
    elif mode == 2:
        simulator = CellMoveSimulator(data)
    elif mode == 3:
        simulator = Simulator2(data)
    else:
        simulator = AdvancedSimulator2()
    simulators[id(simulator)] = simulator
    return {'id': id(simulator)}


@get('/info/<_id:int>')
def get_dungeon_info(_id: int = -1):
    if simulators.get(_id, None) is None:
        return {}
    return simulators[_id].info()


@post('/action/<_id:int>')
def action(_id: int):
    data = request.json
    reward = simulators[_id].action(data['action'])
    info = simulators[_id].info()
    if reward is None:
        info['reward'] = 0
    else:
        info['reward'] = reward
    return info


@post('/reset/<_id:int>')
def reset(_id: int):
    simulators[_id].reset()


@get('/log/list')
def log_list():
    return {
        'list': sorted([f for f in os.listdir('log/') if f[-4:] == '.log'], reverse=True)
    }


@get('/log/<filename>')
def log_file(filename: str):
    return static_file(filename, './log')


if __name__ == '__main__':
    run(host='0.0.0.0', port=8081, debug=True, reloader=True)

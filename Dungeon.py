from enum import Enum
from typing import List

import numpy as np
import random

from Road import Road
from Room import Room, RoomInfo
from util import FOUR_DIRECTION_VECTOR


# 各マスの種類を指す
class CellInfo(Enum):
    WALL = 0
    ROOM = 1
    ROAD = 2
    AGENT = 3
    ENEMY = 4
    GOAL = 5
    PROTECTED = 6
    OTHER = 7


class ColorSequence(Enum):
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    RESET = '\033[0m'


# 各マスの情報をコンソール出力用の色に変換
def cell2color(cell):
    if cell == CellInfo.WALL:
        return ColorSequence.WHITE.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.ROOM:
        return ColorSequence.BLUE.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.ROAD:
        return ColorSequence.CYAN.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.AGENT:
        return ColorSequence.MAGENTA.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.ENEMY:
        return ColorSequence.RED.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.PROTECTED:
        return ColorSequence.YELLOW.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.GOAL:
        return ColorSequence.GREEN.value+'  '+ColorSequence.RESET.value
    elif cell == CellInfo.OTHER:
        return ColorSequence.BLACK.value+'  '+ColorSequence.RESET.value


class Dungeon:

    def __init__(self, row: int, column: int, no_generate_enemy=False):
        """
            row : マップの縦幅(壁)
            column : マップの横幅(壁)
            floor_map : フロアデータ row*column int64
            icon : フロアの各マスの役割のアイコン
                    0 → 壁
                    1 → 空間(歩ける)
            min_div_size : 分割する区画の最小サイズ(1/16)　 
        """

        self.row = row
        self.column = column
        # フロアマップの作成
        self.floor_map: np.ndarray = np.full((self.row, self.column), CellInfo.WALL, dtype=object)
        self.icon = ["■", "　"]
        # 区画の最小サイズ(5分割)
        # self.min_div_size = [row / 5 + 1, column / 5 + 1]
        self.min_div_size = [9, 9]
        self.div_max = 5
        self.div_count = 0

        # 部屋の最小サイズ
        # self.min_room_size = [row / 5 - 2, column / 5 - 2]
        self.min_room_size = [6, 6]
        Room.min_room_size = self.min_room_size
        # self.max_room_size = [row / 2, column / 2]
        self.max_room_size = [15, 20]
        Room.max_room_size = self.max_room_size
        # 部屋
        self.rooms: List[Room] = []
        # 部屋を作る際に必要な区画の座標を保持する変数
        self.room_info: List[RoomInfo] = []
        # 通路
        self.roads: List[Road] = []
        # 通路のid用のindex
        self._road_count = 0

        # 初期化
        self._div_floor(0, 0, row, column)
        self._make_rooms()
        self._print_rooms2map()
        self._connect_rooms()
        self._print_roads2map()
        self.set_protected_area()
        self.goal_room_index = self._generate_goal()
        if not no_generate_enemy:
            self._generate_enemy()
        # self.floor_map[self.floor_map == CellInfo.OTHER] = CellInfo.WALL
        # self.print_floor_map()

    # フロアマップを分割する
    def _div_floor(self, row_s, column_s, row_e, column_e):
        self.div_count += 1
        # 縦に分割
        # 横方向に領域を取得
        try:
            p = random.randint(self.min_div_size[1], column_e - column_s - self.min_div_size[1])
        except ValueError:
            self.room_info.append(RoomInfo(row_s, column_s, row_e, column_e))
            return
        p = p if p % 2 == 0 else p + 1
        # 境界線を描く
        for row in range(row_s, row_e):
            self.floor_map[row][p + column_s] = CellInfo.OTHER

        # 左側が大きければ変わらない
        column_end = p + column_s

        # 領域の右側が大きい
        if p < (column_e - column_s) / 2:
            # 左側は区画が確定する
            self.room_info.append(RoomInfo(row_s, column_s, row_e, column_end))
            # 次の分割用に領域座標を更新
            column_s = p + column_s + 1
            column_end = column_e
        # 左側が大きい
        else:
            # 右側は区画が確定する
            self.room_info.append(RoomInfo(row_s, column_end + 1, row_e, column_e))

        # 縦方向に領域を取得
        try:
            q = random.randint(self.min_div_size[0], row_e - row_s - self.min_div_size[0])
        except ValueError:
            self.room_info.append(RoomInfo(row_s, column_s + 1, row_e, column_end))
            return
        q = q if q % 2 == 0 else q + 1

        # 横方向に境界線を描く
        for column in range(column_s, column_end):
            self.floor_map[q + row_s][column] = CellInfo.OTHER

        # 上側が大きければ変わらない
        row_end = q + row_s
        # 領域の下側が大きい
        if q < (self.row - row_s) / 2:
            # 上側は区画が確定する
            self.room_info.append(RoomInfo(row_s, column_s, row_end, column_end))
            row_s = q + row_s + 1
            row_end = row_e
        else:
            # 下側は区画が確定する
            self.room_info.append(RoomInfo(row_end + 1, column_s, row_e, column_end))

        if self.div_count == self.div_max:
            self.room_info.append(RoomInfo(row_s, column_s, row_end, column_end))
            return
        else:
            self._div_floor(row_s, column_s, row_end, column_end)

    # 区画内に部屋を作る
    def _make_rooms(self):
        for i, room_data in enumerate(self.room_info):
            try:
                self.rooms.append(Room(room_data.top, room_data.left, room_data.bottom, room_data.right, i))
            except ValueError:
                print('Value Error')
                self.print_floor_map()
                continue

    # マップに部屋のマスを描画
    def _print_rooms2map(self):
        for room in self.rooms:
            room.print_to_map(self.floor_map)
            room.protect_corner(self.floor_map)

    # 部屋を通路で繋げる
    def _connect_rooms(self):
        for i in range(len(self.room_info)-1):
            self._create_road(self.room_info[i], self.rooms[i], self.room_info[i+1],  self.rooms[i+1])

            for j in range(i+2, len(self.room_info)):
                is_connected = self._create_road(self.room_info[i], self.rooms[i], self.room_info[j],  self.rooms[j])
                if is_connected:
                    break

    def _create_road(self, room1_info, room1, room2_info, room2):
        road = Road(room1, room1_info, room2, room2_info, self._road_count)
        self._road_count += 1
        if not road.can_connect:
            return False
        self.roads.append(road)
        return True

    def _generate_goal(self):
        room_index = random.randint(0, len(self.rooms)-1)
        goal_room = self.rooms[room_index]
        room_map = self.get_room_map(goal_room)
        index = random.choice(np.where(room_map.reshape(-1) == CellInfo.ROOM)[0])
        y = int((index // room_map.shape[1])) + goal_room.origin[0]
        x = int((index % room_map.shape[1])) + goal_room.origin[1]
        self.goal_position = (x, y)

        self.floor_map[y][x] = CellInfo.GOAL
        return room_index

    def _generate_enemy(self):
        for room in self.rooms:
            room_map = self.get_room_map(room)
            for _ in range(2):
                index = random.choice(np.where(room_map.reshape(-1) == CellInfo.ROOM)[0])
                y = int((index // room_map.shape[1]) + room.origin[0])
                x = int((index % room_map.shape[1]) + room.origin[1])

                # 敵の位置のマスをマップに設定して，その周囲を保護マスにする
                self.floor_map[y][x] = CellInfo.ENEMY
                self._protect_around(x, y)
                room.initial_enemy_positions.append((x, y))

    def get_room_map(self, room):
        return self.floor_map[room.origin[0]:room.origin[0]+room.size[0], room.origin[1]:room.origin[1]+room.size[1]]

    # 指定したマスの周囲4マスを保護マスにする
    def _protect_around(self, x: int, y: int):
        for v in FOUR_DIRECTION_VECTOR:
            if self.floor_map[y+v[1]][x+v[0]] == CellInfo.ROOM:
                self.floor_map[y+v[1]][x+v[0]] = CellInfo.PROTECTED

    # マップに道のマス情報を描画
    def _print_roads2map(self):
        for road in self.roads:
            road.print2map(self.floor_map)

    # マップ情報を標準出力
    def print_floor_map(self):
        for row in self.floor_map:
            for column in row:
                print(cell2color(column), end="")
            print()

    def set_protected_area(self):
        for road in self.roads:
            for road_end_position in road.ends.values():
                for v in FOUR_DIRECTION_VECTOR:
                    x = road_end_position[0]+v[0]
                    y = road_end_position[1]+v[1]
                    if self.floor_map[y][x] == CellInfo.ROOM or self.floor_map[y][x] == CellInfo.PROTECTED:
                        self.floor_map[y][x] = CellInfo.PROTECTED
                        self._protect_around(x, y)
                        break

    def remove_protected_area(self):
        self.floor_map[self.floor_map == CellInfo.PROTECTED] = CellInfo.ROOM

    def scaling(self):
        for row in range(self.row):
            for column in range(self.column):
                if column % 5 == 0 and row % 5 == 0:
                    self.floor_map[row][column] = CellInfo.OTHER

    def dump2json(self):
        return {
            'row': self.row,
            'column': self.column,
            'floor_map': self.get_map_as_list(),
            'rooms': [room.dump2json() for room in self.rooms]
        }

    def get_map_as_list(self):
        return [[e.value for e in line] for line in self.floor_map]


if __name__ == "__main__":
    # row, column は 4の倍数にしてください
    row = 30
    column = 40
    dungeon = Dungeon(row, column)

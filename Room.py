from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import random


class Room:
    min_room_size: Tuple[int, int]
    max_room_size: Tuple[int, int]

    def __init__(self, row_s, column_s, row_e, column_e, _id):
        """
            origin(list) : フロアに対してこの部屋の左上の座標
            size(list): 部屋のサイズ
                [0] : height(row)
                [1] : width(column)
        """
        from Road import Road
        self._make_room(row_s, column_s, row_e, column_e)
        self.id = _id
        self.roads: List[Road] = []
        self.initial_enemy_positions: List[Tuple[int, int]] = []

    def _decide_length(self, min_l, start, end, max_l):
        """
            min_l : 部屋一辺の最小の長さ
            start : 部屋の座標の始まり
            end : 部屋の座標の終わり
            max_l : 一辺の限界
        """
        while 1:
            length = random.randint(min_l, end - start - 1)
            length = length if (end - start - length) % 2 == 0 else length - 1
            if length <= max_l:
                return length

    def _make_room(self, row_s, column_s, row_e, column_e):
        min_row = self.min_room_size[0]
        max_row = self.max_room_size[0]
        min_column = self.min_room_size[1]
        max_column = self.max_room_size[1]

        height = self._decide_length(min_row, row_s, row_e, max_row)
        width = self._decide_length(min_column, column_s, column_e, max_column)

        # +1 は切り上げ計算のため
        center = [((row_e - row_s) - height + 1) // 2, ((column_e - column_s) - width + 1) // 2]

        self.origin = [row_s + center[0], column_s + center[1]]
        self.size = [height, width]

    def protect_corner(self, floor_map: np.ndarray):
        from Dungeon import CellInfo
        # 角にエージェントが生成されないようにする
        floor_map[self.origin[0], self.origin[1]] = CellInfo.PROTECTED
        floor_map[self.origin[0] + self.size[0] - 1, self.origin[1]] = CellInfo.PROTECTED
        floor_map[self.origin[0], self.origin[1] + self.size[1] - 1] = CellInfo.PROTECTED
        floor_map[self.origin[0] + self.size[0] - 1, self.origin[1] + self.size[1] - 1] = CellInfo.PROTECTED

    def print_to_map(self, floor_map: np.ndarray):
        from Dungeon import CellInfo
        floor_map[
                self.origin[0]:self.origin[0] + self.size[0],
                self.origin[1]:self.origin[1] + self.size[1]] = CellInfo.ROOM

    def info(self):
        return {
            'id': self.id,
            'origin': self.origin,
            'size': self.size,
            'roadEnds': [road.ends[self.id] for road in self.roads],
            'roads': [road.info() for road in self.roads]
        }

    def dump2json(self):
        return {
            'id': self.id,
            'origin': self.origin,
            'size': self.size,
            'roads': [road.dump2json() for road in self.roads]
        }


@dataclass
class RoomInfo:
    top: int
    left: int
    bottom: int
    right: int

import random
from typing import Tuple, List
import numpy as np


class Road:
    from Room import Room, RoomInfo

    def __init__(self, room1: Room, room1_info: RoomInfo, room2: Room, room2_info: RoomInfo, _id: int):
        self.connected_rooms = (room1, room2)
        self.connected_rooms_info = (room1_info, room2_info)
        self.cells: List[Tuple[int, int]] = []
        self.id = _id

        self.can_connect = self._can_connect()
        if self.can_connect:
            self._connect()

    def _can_connect(self):
        room1_info, room2_info = self.connected_rooms_info
        if room1_info.top == room2_info.bottom + 1 or room1_info.bottom + 1 == room2_info.top:
            return True
        if room1_info.left == room2_info.right + 1 or room1_info.right + 1 == room2_info.left:
            return True
        return False

    def _connect(self):
        room1, room2 = self.connected_rooms
        room1_info, room2_info = self.connected_rooms_info
        # 上下に接続している
        if room1_info.top == room2_info.bottom + 1 or room1_info.bottom + 1 == room2_info.top:
            xx1 = set(range(room1.origin[1], room1.origin[1] + room1.size[1] - 1))
            xx2 = set(range(room2.origin[1], room2.origin[1] + room2.size[1] - 1))
            for road in room1.roads:
                if road.ends[room1.id][0] in xx1:
                    xx1.remove(road.ends[room1.id][0])
            for road in room2.roads:
                if road.ends[room2.id][0] in xx2:
                    xx2.remove(road.ends[room2.id][0])
            x1 = random.choice(list(xx1))
            x2 = random.choice(list(xx2))
            # room1が上側
            if room1_info.top < room2_info.top:
                y1 = room1.origin[0] + room1.size[0]
                y2 = room2.origin[0] - 1
                # 縦方向に通路を引く
                # room1
                for j in range(y1, room1_info.bottom):
                    self.cells.append((j, x1))
                # room2
                for j in range(room2_info.top, y2 + 1):
                    self.cells.append((j, x2))
                # 縦方向の通路を結ぶ
                for j in range(min(x1, x2), max(x1, x2) + 1):
                    self.cells.append((room1_info.bottom, j))
            # room2が上側
            else:
                y1 = room1.origin[0] - 1
                y2 = room2.origin[0] + room2.size[0]
                # 縦方向に通路を引く
                # room1
                for j in range(room1_info.top, y1 + 1):
                    self.cells.append((j, x1))
                # room2
                for j in range(y2, room2_info.bottom):
                    self.cells.append((j, x2))
                # 縦方向の通路を結ぶ
                for j in range(min(x1, x2), max(x1, x2) + 1):
                    self.cells.append((room2_info.bottom, j))
            # 通路の端の座標を登録
            self.ends = {room1.id: (x1, y1), room2.id: (x2, y2)}

        # 左右に接続している
        if room1_info.left == room2_info.right + 1 or room1_info.right + 1 == room2_info.left:
            yy1 = set(range(room1.origin[0], room1.origin[0] + room1.size[0] - 1))
            yy2 = set(range(room2.origin[0], room2.origin[0] + room2.size[0] - 1))
            for road in room1.roads:
                if road.ends[room1.id][1] in yy1:
                    yy1.remove(road.ends[room1.id][1])
            for road in room2.roads:
                if road.ends[room2.id][1] in yy2:
                    yy2.remove(road.ends[room2.id][1])
            y1 = random.choice(list(yy1))
            y2 = random.choice(list(yy2))
            # room1が左側
            if room1_info.left < room2_info.left:
                x1 = room1.origin[1] + room1.size[1]
                x2 = room2.origin[1] - 1
                # 横方向に通路を引く
                # room1
                for j in range(x1, room1_info.right):
                    self.cells.append((y1, j))
                # room2
                for j in range(room2_info.left, x2 + 1):
                    self.cells.append((y2, j))
                # 横方向の通路を結ぶ
                for j in range(min(y1, y2), max(y1, y2) + 1):
                    self.cells.append((j, room1_info.right))
            # room2が左側
            else:
                x1 = room1.origin[1] - 1
                x2 = room2.origin[1] + room2.size[1]
                # 横方向に通路を引く
                # room1
                for j in range(room1_info.left, x1 + 1):
                    self.cells.append((y1, j))
                # room2
                for j in range(x2, room2_info.right):
                    self.cells.append((y2, j))
                # 横方向の通路を結ぶ
                for j in range(min(y1, y2), max(y1, y2) + 1):
                    self.cells.append((j, room2_info.right))
            # 通路の端の座標を登録
            self.ends = {room1.id: (x1, y1), room2.id: (x2, y2)}
        # 各部屋に通路を登録
        room1.roads.append(self)
        room2.roads.append(self)

    def print2map(self, floor_map: np.ndarray):
        from Dungeon import CellInfo
        for cell in self.cells:
            floor_map[cell] = CellInfo.ROAD

    def info(self):
        return {
            'id': self.id,
            'room1Id': self.connected_rooms[0].id,
            'room2Id': self.connected_rooms[1].id,
            'ends': self.ends,
        }

    def dump2json(self):
        return {
            'id': id(self),
            'room1_id': id(self.connected_rooms[0]),
            'room2_id': id(self.connected_rooms[1]),
        }

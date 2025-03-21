from constants import *
from copy import deepcopy


class Board:
    def __init__(self, size=19):
        self.size = size
        self.grid_size = CELL_SIZE * (size - 1)
        self.margin = (WINDOW_SIZE - self.grid_size) // 2
        self.reset()

    def reset(self):
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.territory = [[None for _ in range(self.size)] for _ in range(self.size)]

    def get_group(self, x, y, visited=None):
        if visited is None:
            visited = set()

        color = self.board[y][x]
        if not color:
            return set()

        group = {(x, y)}
        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                    self.board[new_y][new_x] == color and
                    (new_x, new_y) not in visited):
                group.update(self.get_group(new_x, new_y, visited))

        return group

    def count_liberties(self, group):
        liberties = set()
        for x, y in group:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and 0 <= new_y < self.size and
                        self.board[new_y][new_x] is None):
                    liberties.add((new_x, new_y))
        return len(liberties)

    def calculate_territory(self):
        """Tính toán lãnh thổ của mỗi người chơi"""
        self.territory = [[None for _ in range(self.size)] for _ in range(self.size)]
        visited = set()

        def flood_fill(x, y, visited):
            """Tìm tất cả các điểm trống liên thông và xác định chủ sở hữu"""
            if (x, y) in visited or not (0 <= x < self.size and 0 <= y < self.size):
                return set(), set()

            if self.board[y][x] is not None:
                return set(), {self.board[y][x]}

            territory = {(x, y)}
            visited.add((x, y))
            borders = set()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.size and 0 <= new_y < self.size:
                    new_territory, new_borders = flood_fill(new_x, new_y, visited)
                    territory.update(new_territory)
                    borders.update(new_borders)

            return territory, borders

        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] is None and (x, y) not in visited:
                    territory, borders = flood_fill(x, y, visited)
                    # Nếu vùng chỉ tiếp giáp với một màu, đó là lãnh thổ của màu đó
                    if len(borders) == 1:
                        owner = borders.pop()
                        for tx, ty in territory:
                            self.territory[ty][tx] = owner


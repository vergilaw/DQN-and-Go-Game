import numpy as np
from constants import *
from board import Board
from copy import deepcopy
import pygame

class GoGame:
    def __init__(self):
        # Không khởi tạo self.screen và self.font ở đây
        self.board_size = 19
        self.board = Board(self.board_size)
        self.current_player = 'black'
        self.last_move = None
        self.game_states = []
        self.pass_count = 0
        self.captured = {'black': 0, 'white': 0}
        self.show_score = False

    def get_state(self):
        """Trả về trạng thái bàn cờ dưới dạng ma trận 2D"""
        state = np.zeros((self.board_size, self.board_size), dtype=int)
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board.board[y][x] == 'black':
                    state[y][x] = 1
                elif self.board.board[y][x] == 'white':
                    state[y][x] = -1
        return state

    def get_reward(self, player):
        """Tính phần thưởng cho người chơi"""
        opponent = 'white' if player == 'black' else 'black'

        # Trường hợp trò chơi kết thúc
        if self.show_score:
            scores = self.calculate_score()
            if scores[player] > scores[opponent]:
                return 10
            else:
                return -10

        reward = 0

        # 1. Thưởng/Phạt dựa trên số quân bắt được
        reward += self.captured[player] * 0.5
        reward -= self.captured[opponent] * 0.2

        # 2. Thưởng dựa trên lãnh thổ tiềm năng
        self.board.calculate_territory()
        territory_player = sum(1 for y in range(self.board_size) for x in range(self.board_size)
                               if self.board.territory[y][x] == player)
        territory_opponent = sum(1 for y in range(self.board_size) for x in range(self.board_size)
                                 if self.board.territory[y][x] == opponent)
        reward += territory_player * 0.05
        reward -= territory_opponent * 0.05

        # 3. Thưởng/Phạt dựa trên nước đi cuối cùng
        if self.last_move:
            x, y = self.last_move
            captured_stones = False
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board.size and 0 <= ny < self.board.size and
                        self.board.board[ny][nx] == opponent):
                    group = self.board.get_group(nx, ny)
                    if self.board.count_liberties(group) == 0:
                        captured_stones = True
                        break
            if captured_stones:
                reward += 1.0
            elif self.board.territory[y][x] == player:
                has_opponent_nearby = any(
                    self.board.board[ny][nx] == opponent
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    for nx, ny in [(x + dx, y + dy)]
                    if 0 <= nx < self.board.size and 0 <= ny < self.board.size
                )
                if not has_opponent_nearby:
                    reward -= 0.5
            elif self.board.territory[y][x] == opponent:
                reward += 0.5

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board.size and 0 <= ny < self.board.size and
                        self.board.board[ny][nx] == player):
                    group = self.board.get_group(nx, ny)
                    liberties = self.board.count_liberties(group)
                    if liberties == 1 and (x, y) in [(gx + gdx, gy + gdy)
                                                    for gx, gy in group
                                                    for gdx, gdy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                                    if 0 <= gx + gdx < self.board.size and
                                                       0 <= gy + gdy < self.board.size and
                                                       self.board.board[gy + gdy][gx + gdx] is None]:
                        reward += 0.8

        return reward

    def would_be_suicide(self, x, y):
        """Kiểm tra nước đi có tự sát không"""
        self.board.board[y][x] = self.current_player
        group = self.board.get_group(x, y)
        liberties = self.board.count_liberties(group)
        self.board.board[y][x] = None
        return liberties == 0

    def is_ko(self, x, y):
        """Kiểm tra luật đánh quẩn"""
        if not self.game_states:
            return False
        temp_board = deepcopy(self.board.board)
        temp_board[y][x] = self.current_player
        return any(temp_board == state for state in self.game_states[-8:])

    def is_eye(self, x, y, player):
        """Kiểm tra xem một điểm có phải là mắt không"""
        if self.board.board[y][x] is not None:
            return False
        surrounding_player = 0
        surrounding_total = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board.size and 0 <= ny < self.board.size:
                surrounding_total += 1
                if self.board.board[ny][nx] == player:
                    surrounding_player += 1
        return surrounding_player == surrounding_total and surrounding_total > 0

    def capture_stones(self, x, y):
        """Bắt quân không còn khí"""
        opponent = 'white' if self.current_player == 'black' else 'black'
        captured = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.board.size and 0 <= new_y < self.board.size and
                    self.board.board[new_y][new_x] == opponent):
                group = self.board.get_group(new_x, new_y)
                if self.board.count_liberties(group) == 0:
                    captured.extend(group)
                    for cx, cy in group:
                        self.board.board[cy][cx] = None
        return captured

    def change_board_size(self, size):
        self.board_size = size
        self.board = Board(size)
        self.reset_game()

    def reset_game(self):
        self.board.reset()
        self.current_player = 'black'
        self.last_move = None
        self.game_states = []
        self.pass_count = 0
        self.captured = {'black': 0, 'white': 0}
        self.show_score = False

    def calculate_score(self):
        """Tính điểm cuối game"""
        self.board.calculate_territory()
        scores = {'black': self.captured['black'], 'white': self.captured['white']}
        for y in range(self.board.size):
            for x in range(self.board.size):
                if self.board.territory[y][x]:
                    scores[self.board.territory[y][x]] += 1
        return scores

    def make_move(self, pos):
        if pos == 'pass':
            self.pass_count += 1
            if self.pass_count >= 2:
                self.show_score = True
            self.current_player = 'white' if self.current_player == 'black' else 'black'
            return True

        x, y = pos
        if self.board.board[y][x] is not None:
            return False
        if self.would_be_suicide(x, y) or self.is_ko(x, y):
            return False

        self.game_states.append(deepcopy(self.board.board))
        if len(self.game_states) > 8:
            self.game_states.pop(0)

        self.board.board[y][x] = self.current_player
        self.last_move = (x, y)
        captured = self.capture_stones(x, y)
        if captured:
            self.captured[self.current_player] += len(captured)

        self.pass_count = 0
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        return True

    def draw(self, screen, font):
        """Vẽ trò chơi lên màn hình, sử dụng font được truyền vào"""
        screen.fill(BROWN)

        for i in range(self.board.size):
            pygame.draw.line(screen, BLACK,
                             (self.board.margin, self.board.margin + i * CELL_SIZE),
                             (self.board.margin + self.board.grid_size, self.board.margin + i * CELL_SIZE))
            pygame.draw.line(screen, BLACK,
                             (self.board.margin + i * CELL_SIZE, self.board.margin),
                             (self.board.margin + i * CELL_SIZE, self.board.margin + self.board.grid_size))

        if self.board.size == 19:
            hoshi_points = [3, 9, 15]
        elif self.board.size == 13:
            hoshi_points = [3, 6, 9]
        else:
            hoshi_points = [2, 4, 6]

        for x in hoshi_points:
            for y in hoshi_points:
                pygame.draw.circle(screen, BLACK,
                                   (self.board.margin + x * CELL_SIZE,
                                    self.board.margin + y * CELL_SIZE), 5)

        for y in range(self.board.size):
            for x in range(self.board.size):
                pos_x = self.board.margin + x * CELL_SIZE
                pos_y = self.board.margin + y * CELL_SIZE
                if self.board.board[y][x]:
                    color = BLACK if self.board.board[y][x] == 'black' else WHITE
                    pygame.draw.circle(screen, color, (pos_x, pos_y), 18)
                elif self.show_score and self.board.territory[y][x]:
                    color = (0, 0, 128) if self.board.territory[y][x] == 'black' else (255, 192, 203)
                    pygame.draw.rect(screen, color,
                                     (pos_x - CELL_SIZE // 2, pos_y - CELL_SIZE // 2,
                                      CELL_SIZE, CELL_SIZE), 0)

        if self.last_move:
            x, y = self.last_move
            pygame.draw.circle(screen, RED,
                               (self.board.margin + x * CELL_SIZE,
                                self.board.margin + y * CELL_SIZE), 5)

        current_text = f"Turn: {'Black' if self.current_player == 'black' else 'White'}"
        text_surface = font.render(current_text, True, BLACK)
        screen.blit(text_surface, (10, 10))

        captured_text = f"Capture - Black: {self.captured['black']}, White: {self.captured['white']}"
        text_surface = font.render(captured_text, True, BLACK)
        screen.blit(text_surface, (10, 40))

        if self.show_score:
            scores = self.calculate_score()
            score_text = f"Points - Black: {scores['black']}, White: {scores['white']}"
            text_surface = font.render(score_text, True, RED)
            screen.blit(text_surface, (WINDOW_SIZE // 2 - 150, 10))
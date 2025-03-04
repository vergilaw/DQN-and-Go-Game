import numpy as np

from constants import *
from board import Board
from copy import deepcopy


class GoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Cờ Vây - Go Game')
        self.font = pygame.font.Font(None, 36)

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
        if self.show_score:
            scores = self.calculate_score()
            opponent = 'white' if player == 'black' else 'black'
            if scores[player] > scores[opponent]:
                return 10
            else:
                return -10

        reward = 0
        opponent = 'white' if player == 'black' else 'black'

        # Thưởng khi bắt được quân
        if self.captured[player] > 0:
            reward += self.captured[player] * 0.5

        # Phạt khi bị bắt quân
        if self.captured[opponent] > 0:
            reward -= self.captured[opponent] * 0.1

        # Phạt khi đánh vào lãnh thổ đã chiếm
        if self.last_move:
            x, y = self.last_move

            # Kiểm tra xem nước đi này có ăn được quân đối phương không
            captured_stones = False
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board.size and 0 <= ny < self.board.size and
                        self.board.board[ny][nx] == opponent):
                    group = self.board.get_group(nx, ny)
                    if self.board.count_liberties(group) == 0:
                        captured_stones = True
                        break

            # Nếu nước đi này ăn được quân đối phương, thưởng lớn
            if captured_stones:
                reward += 1.0
            # Ngược lại, nếu đi vào lãnh thổ của chính mình mà không ăn được quân
            elif self.board.territory[y][x] == player:
                # Kiểm tra xem có quân đối phương xung quanh không
                has_opponent_nearby = False
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.board.size and 0 <= ny < self.board.size and
                            self.board.board[ny][nx] == opponent):
                        has_opponent_nearby = True
                        break

                # Nếu không có quân đối phương xung quanh, phạt vì đi vào lãnh thổ không cần thiết
                if not has_opponent_nearby:
                    reward -= 1.0
                # Nếu có quân đối phương xung quanh, thưởng vì bảo vệ lãnh thổ
                else:
                    reward += 0.2

        # chặn đối phương xây dựng lãnh thổ
        if self.last_move:
            x, y = self.last_move
            if self.board.territory[y][x] == opponent:
                reward += 0.3

        # Phần thưởng khi bảo vệ quân của mình khỏi bị
        if self.last_move:
            x, y = self.last_move
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board.size and 0 <= ny < self.board.size and
                        self.board.board[ny][nx] == player):
                    group = self.board.get_group(nx, ny)
                    liberties = self.board.count_liberties(group)
                    if liberties == 1:
                        # last chi check
                        for gx, gy in group:
                            for gdx, gdy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                liberty_x, liberty_y = gx + gdx, gy + gdy
                                if (0 <= liberty_x < self.board.size and
                                        0 <= liberty_y < self.board.size and
                                        self.board.board[liberty_y][liberty_x] is None and
                                        (liberty_x, liberty_y) == (x, y)):
                                    reward += 0.4  # Thưởng khi cứu quân của mình khỏi bị ăn
                                    break

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
        """Kiểm tra xem một điểm có phải là mắt (eye) không"""
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

        # check eye
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

        # Đếm lãnh thổ
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

        # Lưu trạng thái và đặt quân
        self.game_states.append(deepcopy(self.board.board))
        if len(self.game_states) > 8:
            self.game_states.pop(0)

        self.board.board[y][x] = self.current_player
        self.last_move = (x, y)

        # Bắt quân
        captured = self.capture_stones(x, y)
        if captured:
            self.captured[self.current_player] += len(captured)

        self.pass_count = 0
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        return True

    def draw(self):
        self.screen.fill(BROWN)

        # Vẽ lưới
        for i in range(self.board.size):
            pygame.draw.line(self.screen, BLACK,
                             (self.board.margin, self.board.margin + i * CELL_SIZE),
                             (self.board.margin + self.board.grid_size, self.board.margin + i * CELL_SIZE))
            pygame.draw.line(self.screen, BLACK,
                             (self.board.margin + i * CELL_SIZE, self.board.margin),
                             (self.board.margin + i * CELL_SIZE, self.board.margin + self.board.grid_size))

        # Vẽ hoshi
        if self.board.size == 19:
            hoshi_points = [3, 9, 15]
        elif self.board.size == 13:
            hoshi_points = [3, 6, 9]
        else:  # size 9
            hoshi_points = [2, 4, 6]

        for x in hoshi_points:
            for y in hoshi_points:
                pygame.draw.circle(self.screen, BLACK,
                                   (self.board.margin + x * CELL_SIZE,
                                    self.board.margin + y * CELL_SIZE), 5)

        # Vẽ quân cờ và lãnh thổ
        for y in range(self.board.size):
            for x in range(self.board.size):
                pos_x = self.board.margin + x * CELL_SIZE
                pos_y = self.board.margin + y * CELL_SIZE

                if self.board.board[y][x]:
                    color = BLACK if self.board.board[y][x] == 'black' else WHITE
                    pygame.draw.circle(self.screen, color, (pos_x, pos_y), 18)
                elif self.show_score and self.board.territory[y][x]:
                    color = (0, 0, 128) if self.board.territory[y][x] == 'black' else (255, 192, 203)
                    pygame.draw.rect(self.screen, color,
                                     (pos_x - CELL_SIZE // 2, pos_y - CELL_SIZE // 2,
                                      CELL_SIZE, CELL_SIZE), 0)


        if self.last_move:
            x, y = self.last_move
            pygame.draw.circle(self.screen, RED,
                               (self.board.margin + x * CELL_SIZE,
                                self.board.margin + y * CELL_SIZE), 5)

        # Hiển thị thông tin game
        current_text = f"Turn: {'Black' if self.current_player == 'black' else 'White'}"
        text_surface = self.font.render(current_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))

        captured_text = f"Capture - Black: {self.captured['black']}, White: {self.captured['white']}"
        text_surface = self.font.render(captured_text, True, BLACK)
        self.screen.blit(text_surface, (10, 40))

        if self.show_score:
            scores = self.calculate_score()
            score_text = f"Points - Black: {scores['black']}, White: {scores['white']}"
            text_surface = self.font.render(score_text, True, RED)
            self.screen.blit(text_surface, (WINDOW_SIZE // 2 - 150, 10))

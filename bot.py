import random
import math
import time
from copy import deepcopy


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_legal_moves()

    def get_legal_moves(self):
        # Lấy tất cả các nước đi hợp lệ từ trạng thái hiện tại
        moves = []
        board, current_player = self.game_state
        for y in range(len(board)):
            for x in range(len(board)):
                if board[y][x] is None and not self.would_be_suicide(x, y, board, current_player):
                    moves.append((x, y))

        # Thêm lựa chọn "pass"
        moves.append('pass')
        return moves

    def would_be_suicide(self, x, y, board, color):
        size = len(board)
        temp_board = deepcopy(board)
        temp_board[y][x] = color

        if self.has_liberty(x, y, temp_board):
            return False

        opponent = 'white' if color == 'black' else 'black'
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and temp_board[ny][nx] == opponent:
                group = self.get_group(nx, ny, temp_board)
                if not self.has_liberty_group(group, temp_board):
                    return False

        return True

    def has_liberty(self, x, y, board):
        size = len(board)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
                return True
        return False

    def get_group(self, x, y, board, visited=None):
        if visited is None:
            visited = set()

        size = len(board)
        color = board[y][x]
        if not color:
            return set()

        group = {(x, y)}
        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < size and 0 <= ny < size and
                    board[ny][nx] == color and
                    (nx, ny) not in visited):
                group.update(self.get_group(nx, ny, board, visited))

        return group

    def has_liberty_group(self, group, board):
        # Kiểm tra nếu nhóm quân có khí
        size = len(board)
        for x, y in group:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
                    return True
        return False

    def select_child(self):
        # Chọn nút con theo UCB1
        c = 1.4  # Tham số cân bằng giữa khám phá và khai thác
        s = max(child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits)
                for child in self.children)
        return random.choice([child for child in self.children if
                              child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits) == s])

    def add_child(self, move, game_state):
        # Thêm nút con mới
        child = MCTSNode(game_state, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        # Cập nhật thông tin nút
        self.visits += 1
        self.wins += result


class MCTS:
    def __init__(self, time_limit=1.0):
        self.time_limit = time_limit

    def get_move(self, game):
        # Lấy nước đi tốt nhất từ MCTS
        root = MCTSNode((deepcopy(game.board.board), game.current_player))

        end_time = time.time() + self.time_limit

        # Nếu không còn nước đi hợp lệ, pass
        if not root.untried_moves or (len(root.untried_moves) == 1 and root.untried_moves[0] == 'pass'):
            return 'pass'

        # Kiểm tra xem có nên pass không dựa trên trạng thái bàn cờ
        board_filled_ratio = self.calculate_board_filled_ratio(game.board.board)
        # Nếu bàn cờ đã lấp đầy > 80% và đã qua 100 nước, tăng khả năng pass
        should_consider_pass = board_filled_ratio > 0.8 and game.pass_count > 0

        while time.time() < end_time:
            # Chọn
            node = root

            # Tạo bản sao của trạng thái game thay vì toàn bộ đối tượng game
            board_copy = deepcopy(game.board.board)
            current_player = game.current_player

            # Chọn nút tốt nhất cho đến khi gặp nút chưa mở rộng hoàn toàn
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                if node.move != 'pass':
                    x, y = node.move
                    # Cập nhật bản sao của bàn cờ
                    board_copy[y][x] = current_player
                    # Đổi người chơi
                    current_player = 'white' if current_player == 'black' else 'black'

            # Mở rộng
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                # Cập nhật bản sao của bàn cờ
                if move != 'pass':
                    x, y = move
                    board_copy[y][x] = current_player

                # Đổi người chơi
                next_player = 'white' if current_player == 'black' else 'black'

                # Thêm nút con mới
                child = MCTSNode((deepcopy(board_copy), next_player), node, move)
                node.children.append(child)
                node = child

            # Mô phỏng (simulation)
            # Thực hiện các nước đi ngẫu nhiên cho đến khi kết thúc game
            sim_board = deepcopy(board_copy)
            sim_player = current_player
            sim_pass_count = 0
            max_moves = 30  # Giới hạn số nước mô phỏng để tránh vòng lặp vô hạn

            for _ in range(max_moves):
                if sim_pass_count >= 2:
                    break

                # Lấy các nước đi hợp lệ
                valid_moves = []
                for y in range(len(sim_board)):
                    for x in range(len(sim_board)):
                        if sim_board[y][x] is None:
                            # Kiểm tra nếu nước đi không phải tự sát
                            temp_board = deepcopy(sim_board)
                            temp_board[y][x] = sim_player
                            if self.has_liberty(x, y, temp_board):
                                valid_moves.append((x, y))

                # Thêm lựa chọn pass
                valid_moves.append('pass')

                # Chọn nước đi ngẫu nhiên
                move = random.choice(valid_moves)

                if move == 'pass':
                    sim_pass_count += 1
                else:
                    x, y = move
                    sim_board[y][x] = sim_player
                    sim_pass_count = 0

                    # Bắt quân không còn khí
                    self.capture_stones(x, y, sim_board, sim_player)

                # Đổi người chơi
                sim_player = 'white' if sim_player == 'black' else 'black'

            # Đánh giá kết quả
            result = self.evaluate_board(sim_board, game.current_player)

            # Cập nhật (backpropagation)
            while node is not None:
                node.update(result)
                node = node.parent

        # Chọn nước đi tốt nhất dựa trên số lần thăm
        if root.children:
            # Sắp xếp các nước đi theo số lần thăm
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
            best_child = sorted_children[0]

            # Tìm nút pass nếu có
            pass_nodes = [c for c in root.children if c.move == 'pass']

            # Nếu đang ở giai đoạn cuối game và điểm số của nước đi tốt nhất không cao
            if should_consider_pass and best_child.wins / best_child.visits < 0.4:
                # Nếu có nút pass và tỷ lệ thắng của nó không quá thấp
                if pass_nodes and pass_nodes[0].visits > root.visits * 0.1:
                    return 'pass'

            # Nếu không còn nước đi nào tốt (tỷ lệ thắng thấp)
            if best_child.wins / best_child.visits < 0.2 and pass_nodes:
                return 'pass'

            return best_child.move

        return 'pass'  # No move = pass

    def calculate_board_filled_ratio(self, board):
        """Tính tỷ lệ bàn cờ đã được lấp đầy"""
        total_cells = len(board) * len(board)
        filled_cells = sum(1 for row in board for cell in row if cell is not None)
        return filled_cells / total_cells

    def has_liberty(self, x, y, board):
        """Kiểm tra xem quân cờ có khí không"""
        size = len(board)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
                return True
        return False

    def get_group(self, x, y, board, visited=None):
        """Lấy nhóm quân cùng màu liên kết với quân tại (x,y)"""
        if visited is None:
            visited = set()

        size = len(board)
        color = board[y][x]
        if not color:
            return set()

        group = {(x, y)}
        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < size and 0 <= ny < size and
                    board[ny][nx] == color and
                    (nx, ny) not in visited):
                group.update(self.get_group(nx, ny, board, visited))

        return group

    def capture_stones(self, x, y, board, player):
        """Bắt quân không còn khí"""
        size = len(board)
        opponent = 'white' if player == 'black' else 'black'

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and board[ny][nx] == opponent:
                group = self.get_group(nx, ny, board)
                if not self.has_liberty_group(group, board):
                    # Bắt quân
                    for gx, gy in group:
                        board[gy][gx] = None

    def has_liberty_group(self, group, board):
        """Kiểm tra xem nhóm quân có khí không"""
        size = len(board)
        for x, y in group:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
                    return True
        return False

    def evaluate_board(self, board, player):
        """Đánh giá bàn cờ cho người chơi hiện tại"""
        # Đếm số quân của mỗi người chơi
        black_count = sum(1 for row in board for cell in row if cell == 'black')
        white_count = sum(1 for row in board for cell in row if cell == 'white')

        # Tính điểm theo tỷ lệ quân
        if player == 'black':
            return 1 if black_count >= white_count else 0
        else:
            return 1 if white_count >= black_count else 0

    def is_game_over(self, game):
        #pass >=2 end
        return game.pass_count >= 2

    def get_result(self, game, player):
        scores = game.calculate_score()
        if scores['black'] > scores['white']:
            return 1 if player == 'black' else 0
        elif scores['white'] > scores['black']:
            return 1 if player == 'white' else 0
        else:
            return 0.5  # Hòa

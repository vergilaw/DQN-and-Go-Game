# import random
# import math
# import time
# import numpy as np
# from threading import Thread, Lock
# from copy import deepcopy
#
#
# class MCTSNode:
#     def __init__(self, game_state, parent=None, move=None):
#         self.game_state = (np.copy(game_state[0]), game_state[1])  # board, current_player
#         self.parent = parent
#         self.move = move
#         self.children = []
#         self.wins = 0
#         self.visits = 0
#         # Lưu danh sách ô trống để giảm thời gian quét
#         self.empty_cells = set((x, y) for y in range(len(game_state[0]))
#                               for x in range(len(game_state[0]))
#                               if game_state[0][y][x] is None)
#         self.untried_moves = self.get_legal_moves()
#
#     def get_legal_moves(self):
#         moves = []
#         board, current_player = self.game_state
#         for x, y in self.empty_cells:
#             if not self.would_be_suicide(x, y, board, current_player):
#                 moves.append((x, y))
#         moves.append('pass')
#         return moves
#
#     def would_be_suicide(self, x, y, board, color):
#         size = len(board)
#         temp_board = np.copy(board)
#         temp_board[y][x] = color
#
#         if self.has_liberty(x, y, temp_board):
#             return False
#
#         opponent = 'white' if color == 'black' else 'black'
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < size and 0 <= ny < size and temp_board[ny][nx] == opponent:
#                 group = self.get_group(nx, ny, temp_board)
#                 if not self.has_liberty_group(group, temp_board):
#                     return False
#         return True
#
#     def has_liberty(self, x, y, board):
#         size = len(board)
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
#                 return True
#         return False
#
#     def get_group(self, x, y, board):
#         size = len(board)
#         color = board[y][x]
#         if not color:
#             return set()
#         group = set()
#         stack = [(x, y)]
#         while stack:
#             cx, cy = stack.pop()
#             if (cx, cy) in group:
#                 continue
#             group.add((cx, cy))
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 nx, ny = cx + dx, cy + dy
#                 if (0 <= nx < size and 0 <= ny < size and
#                         board[ny][nx] == color and (nx, ny) not in group):
#                     stack.append((nx, ny))
#         return group
#
#     def has_liberty_group(self, group, board):
#         size = len(board)
#         for x, y in group:
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
#                     return True
#         return False
#
#     def select_child(self):
#         # Nếu node cha chưa được thăm, chọn ngẫu nhiên một child
#         if self.visits == 0:
#             return random.choice(self.children)
#
#         # Điều chỉnh c dựa trên số lần thăm
#         c = 1.4 if self.visits < 100 else 0.7
#
#         # Tính UCB1, xử lý trường hợp child.visits == 0
#         def ucb1(child):
#             if child.visits == 0:
#                 return float('inf')  # Ưu tiên thăm các node chưa được khám phá
#             return child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits)
#
#         s = max(ucb1(child) for child in self.children)
#         return random.choice([child for child in self.children if ucb1(child) == s])
#
#     def add_child(self, move, game_state):
#         child = MCTSNode(game_state, self, move)
#         self.untried_moves.remove(move)
#         self.children.append(child)
#         if move != 'pass':
#             self.empty_cells.remove(move)  # Cập nhật ô trống
#         return child
#
#     def update(self, result):
#         self.visits += 1
#         self.wins += result
#
#
# class MCTS:
#     def __init__(self, time_limit=1.0):
#         self.time_limit = time_limit
#         self.state_cache = {}  # Bộ nhớ đệm trạng thái
#
#     def get_move(self, game):
#         start_time = time.time()
#         root = MCTSNode((np.copy(game.board.board), game.current_player))
#         if not root.untried_moves or root.untried_moves == ['pass']:
#             return 'pass'
#
#         board_filled_ratio = self.calculate_board_filled_ratio(game.board.board)
#         should_consider_pass = board_filled_ratio > 0.8 and game.pass_count > 0
#
#         end_time = time.time() + self.time_limit
#         lock = Lock()
#
#         while time.time() < end_time:
#             node = root
#             board = np.copy(root.game_state[0])
#             player = root.game_state[1]
#
#             # Selection
#             while not node.untried_moves and node.children:
#                 node = node.select_child()
#                 if node.move != 'pass':
#                     x, y = node.move
#                     board[y][x] = player
#                     player = 'white' if player == 'black' else 'black'
#
#             # Expansion
#             if node.untried_moves:
#                 move = random.choice(node.untried_moves)
#                 if move != 'pass':
#                     x, y = move
#                     board[y][x] = player
#                 next_player = 'white' if player == 'black' else 'black'
#                 child = node.add_child(move, (np.copy(board), next_player))
#
#                 # Simulation (song song)
#                 threads = []
#                 for _ in range(4):  # 4 luồng
#                     t = Thread(target=self.run_simulation, args=(child, np.copy(board), next_player, lock))
#                     threads.append(t)
#                     t.start()
#                 for t in threads:
#                     t.join()
#
#         # Chọn nước đi tốt nhất
#         if root.children:
#             sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
#             best_child = sorted_children[0]
#             pass_nodes = [c for c in root.children if c.move == 'pass']
#
#             if should_consider_pass and best_child.wins / best_child.visits < 0.4:
#                 if pass_nodes and pass_nodes[0].visits > root.visits * 0.1:
#                     return 'pass'
#             if best_child.wins / best_child.visits < 0.2 and pass_nodes:
#                 return 'pass'
#             return best_child.move
#
#         print(f"Time taken: {time.time() - start_time:.2f}s")
#         return 'pass'
#
#     def run_simulation(self, node, board, player, lock):
#         sim_board = np.copy(board)
#         sim_player = player
#         sim_pass_count = 0
#         max_moves = 30
#
#         for _ in range(max_moves):
#             if sim_pass_count >= 2:
#                 break
#
#             valid_moves = [(x, y) for x, y in node.empty_cells if self.has_liberty(x, y, sim_board)]
#             valid_moves.append('pass')
#
#             move = random.choice(valid_moves)
#             if move == 'pass':
#                 sim_pass_count += 1
#             else:
#                 x, y = move
#                 sim_board[y][x] = sim_player
#                 sim_pass_count = 0
#                 self.capture_stones(x, y, sim_board, sim_player)
#             sim_player = 'white' if sim_player == 'black' else 'black'
#
#         result = self.evaluate_board(sim_board, node.game_state[1])
#         with lock:
#             node.update(result)
#
#     def calculate_board_filled_ratio(self, board):
#         total_cells = len(board) * len(board)
#         filled_cells = np.count_nonzero(board != None)
#         return filled_cells / total_cells
#
#     def capture_stones(self, x, y, board, player):
#         size = len(board)
#         opponent = 'white' if player == 'black' else 'black'
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < size and 0 <= ny < size and board[ny][nx] == opponent:
#                 group = self.get_group(nx, ny, board)
#                 if not self.has_liberty_group(group, board):
#                     for gx, gy in group:
#                         board[gy][gx] = None
#
#     def has_liberty(self, x, y, board):
#         size = len(board)
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
#                 return True
#         return False
#
#     def get_group(self, x, y, board):
#         size = len(board)
#         color = board[y][x]
#         if not color:
#             return set()
#         group = set()
#         stack = [(x, y)]
#         while stack:
#             cx, cy = stack.pop()
#             if (cx, cy) in group:
#                 continue
#             group.add((cx, cy))
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 nx, ny = cx + dx, cy + dy
#                 if (0 <= nx < size and 0 <= ny < size and
#                         board[ny][nx] == color and (nx, ny) not in group):
#                     stack.append((nx, ny))
#         return group
#
#     def has_liberty_group(self, group, board):
#         size = len(board)
#         for x, y in group:
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < size and 0 <= ny < size and board[ny][nx] is None:
#                     return True
#         return False
#
#     def evaluate_board(self, board, player):
#         board_tuple = tuple(map(tuple, board))
#         if board_tuple in self.state_cache:
#             return self.state_cache[board_tuple]
#
#         black_count = np.count_nonzero(board == 'black')
#         white_count = np.count_nonzero(board == 'white')
#         center_size = len(board) // 3
#         center_control = np.count_nonzero(board[center_size:2*center_size, center_size:2*center_size] == player)
#         score = (black_count if player == 'black' else white_count) + center_control * 0.1
#         opponent_score = (white_count if player == 'black' else black_count)
#         result = 1 if score > opponent_score else 0
#         self.state_cache[board_tuple] = result
#         return result
#
#
# # Ví dụ sử dụng (giả định lớp Game)
# class DummyGame:
#     def __init__(self, size=5):
#         self.board = type('Board', (), {'board': np.full((size, size), None)})
#         self.current_player = 'black'
#         self.pass_count = 0
#
#     def calculate_score(self):
#         return {'black': 0, 'white': 0}
#
#
# if __name__ == "__main__":
#     game = DummyGame(size=5)
#     mcts = MCTS(time_limit=1.0)
#     move = mcts.get_move(game)
#     print(f"Best move: {move}")
# import torch
# print("PyTorch version:", torch.__version__)
# print("GPU available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

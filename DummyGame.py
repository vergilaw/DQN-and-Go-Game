# class DummyGame:
#     def __init__(self, board_size):
#         self.board_size = board_size
#         self.board = np.zeros((board_size, board_size))
#         self.is_over_flag = False
#
#     def apply_action(self, action):
#         # Giả lập hành động (cần thay bằng logic Go thực tế)
#         reward = 1 if not self.is_over_flag else 10
#         self.is_over_flag = random.random() < 0.1  # Giả lập kết thúc ngẫu nhiên
#         return reward
#
#     def is_over(self):
#         return self.is_over_flag
#
#
# def train_bot_vs_bot(episodes, board_size=5):
#     """Huấn luyện hai bot đấu với nhau"""
#     agent1 = DQNAgent(board_size)
#     agent2 = DQNAgent(board_size)
#
#     for e in range(episodes):
#         game = DummyGame(board_size)
#         state = np.reshape(game.board, [1, board_size, board_size, 1])
#         done = False
#
#         while not done:
#             # Lượt của Agent1
#             action1 = agent1.act(state)
#             reward1 = game.apply_action(action1)
#             next_state = np.reshape(game.board, [1, board_size, board_size, 1])
#             done = game.is_over()
#             agent1.remember(state, action1, reward1, next_state, done)
#             state = next_state
#
#             if not done:
#                 # Lượt của Agent2
#                 action2 = agent2.act(state)
#                 reward2 = game.apply_action(action2)
#                 next_state = np.reshape(game.board, [1, board_size, board_size, 1])
#                 done = game.is_over()
#                 agent2.remember(state, action2, reward2, next_state, done)
#                 state = next_state
#
#         # Huấn luyện sau mỗi ván
#         agent1.replay(32)
#         agent2.replay(32)
#         agent1.update_target_model()
#         agent2.update_target_model()
#
#         print(f"Episode {e + 1}/{episodes} completed")
#
#     # Lưu model sau khi huấn luyện
#     agent1.save_model('agent1_model.h5')
#     agent2.save_model('agent2_model.h5')
#
#
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

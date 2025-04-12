import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import math
import copy
from threading import Lock
import pickle
import matplotlib.pyplot as plt


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior if prior is not None else 0.0  # Đảm bảo prior không là None

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = None
        for action, child in self.children.items():
            prior = child.prior if child.prior is not None else 0.0  # Kiểm tra prior
            exploration = (c_puct * prior * math.sqrt(self.visits) / (1 + child.visits)
                           if self.visits > 0 else 0)
            score = child.value() + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, self.children.get(best_action, None)  # Trả về None nếu không có best_action

    def expand(self, actions, action_priors):
        for action, prior in zip(actions, action_priors):
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=None, parent=self, action=action, prior=prior
                )

    def update(self, value):
        self.visits += 1
        self.value_sum += value


class MCTS:
    def __init__(self, model, board_size, num_simulations=50, c_puct=1.0, temperature=1.0):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.game = None

    def search(self, state, game):
        self.game = game
        root = MCTSNode(state)
        valid_actions = self._get_valid_actions(state, game)
        if not valid_actions:
            return self.board_size * self.board_size, np.array([1.0])
        action_priors = self._get_action_priors(state, valid_actions)
        # Kiểm tra action_priors
        if action_priors is None or len(action_priors) != len(valid_actions):
            action_priors = np.ones(len(valid_actions)) / len(valid_actions)  # Fallback
        root.expand(valid_actions, action_priors)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            game_copy = self._copy_game(game)
            current_player = game.current_player
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                if action is None or node is None:  # Nếu không chọn được hành động
                    break
                self._apply_action(game_copy, action)
                search_path.append(node)
            game_over, value = self._is_terminal(game_copy)
            if not game_over:
                next_state = self._get_state(game_copy)
                valid_actions = self._get_valid_actions(next_state, game_copy)
                if valid_actions:
                    action_priors = self._get_action_priors(next_state, valid_actions)
                    if action_priors is None or len(action_priors) != len(valid_actions):
                        action_priors = np.ones(len(valid_actions)) / len(valid_actions)
                    node.expand(valid_actions, action_priors)
                    value = self._evaluate(next_state)
                else:
                    value = 0
            for node in reversed(search_path):
                node.update(value if current_player == game.current_player else -value)
        actions = list(root.children.keys())
        if not actions:  # Nếu không có hành động nào
            return self.board_size * self.board_size, np.array([1.0])
        visits = [root.children[action].visits for action in actions]
        if self.temperature == 0:
            action_idx = np.argmax(visits)
            probs = np.zeros(len(actions))
            probs[action_idx] = 1.0
        else:
            visits = np.array(visits) ** (1.0 / self.temperature)
            probs = visits / np.sum(visits)
        return actions[np.argmax(probs)], probs

    def _get_valid_actions(self, state, game):
        valid_actions = [self.board_size * self.board_size]
        for y in range(self.board_size):
            for x in range(self.board_size):
                action = y * self.board_size + x
                if (game.board.board[y][x] is None and
                        not game.would_be_suicide(x, y) and
                        not game.is_ko(x, y)):
                    valid_actions.append(action)
        return valid_actions

    def _copy_game(self, game):
        return copy.deepcopy(game)

    def _apply_action(self, game, action):
        if action == self.board_size * self.board_size:
            game.make_move('pass')
        else:
            x, y = divmod(action, self.board_size)
            game.make_move((x, y))

    def _get_state(self, game):
        return game.get_state()

    def _is_terminal(self, game):
        if game.pass_count >= 2:
            scores = game.calculate_score()
            if scores['black'] > scores['white']:
                return True, 1 if game.current_player == 'black' else -1
            elif scores['white'] > scores['black']:
                return True, 1 if game.current_player == 'white' else -1
            else:
                return True, 0
        return False, 0

    def _get_action_priors(self, state, valid_actions):
        state_tensor = self._prepare_state(state)
        try:
            q_values = self.model.predict(state_tensor, verbose=0)[0]
            valid_q_values = [q_values[action] for action in valid_actions]
            return self._softmax(valid_q_values)
        except Exception as e:
            print(f"Error in _get_action_priors: {e}")
            return np.ones(len(valid_actions)) / len(valid_actions)  # Fallback

    def _softmax(self, x):
        x = np.array(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _prepare_state(self, state):
        if isinstance(state, np.ndarray):
            if len(state.shape) == 2:
                return np.expand_dims(state, axis=(0, -1))
            elif len(state.shape) == 3:
                return np.expand_dims(state, axis=0)
        return state

    def _evaluate(self, state):
        state_tensor = self._prepare_state(state)
        try:
            q_values = self.model.predict(state_tensor, verbose=0)[0]
            return np.max(q_values)
        except Exception as e:
            print(f"Error in _evaluate: {e}")
            return 0.0  # Fallback


class DQNAgent:
    def __init__(self, board_size=19, model_path=None, use_mcts=True, mcts_simulations=50):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        self.memory = deque(maxlen=5000)
        self.memory_lock = Lock()
        self.model_lock = Lock()
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 500
        self.step_counter = 0
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

        # Thêm các danh sách để lưu dữ liệu huấn luyện
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {gpus}")
        else:
            print("No GPU detected, running on CPU")

        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}. Creating new model.")
                self.model = self._build_model()
        else:
            print(f"No model found at {model_path or 'default path'}. Creating new model.")
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if self.use_mcts:
            self.mcts = MCTS(
                model=self.model,
                board_size=self.board_size,
                num_simulations=self.mcts_simulations,
                c_puct=2.0,
                temperature=1.0
            )

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=[
                tf.TensorSpec(shape=[None, self.board_size, self.board_size, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, self.action_size], dtype=tf.float32)
            ],
            reduce_retracing=True
        )

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.board_size, self.board_size, 1)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        with self.model_lock:
            self.target_model.set_weights(self.model.get_weights())
        print(f"Target model updated. Current epsilon: {self.epsilon:.4f}")

    def remember(self, state, action, reward, next_state, done):
        for obj in (state, action, reward, next_state, done):
            try:
                pickle.dumps(obj)
            except Exception as e:
                print(f"Cannot pickle object: {obj}, type: {type(obj)}, error: {e}")
        with self.memory_lock:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, game=None):
        if self.use_mcts and game and np.random.random() > self.epsilon:
            self.mcts.game = game
            action, _ = self.mcts.search(state[0, :, :, 0], game)
            return action if action is not None else self.board_size * self.board_size  # Trả về pass nếu action là None
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        with self.model_lock:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.ensure_shape(state_tensor, [1, self.board_size, self.board_size, 1])
            act_values = self.model(state_tensor, training=False)
        return np.argmax(act_values.numpy()[0])

    def _train_step(self, states, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss_fn = tf.keras.losses.MeanSquaredError()
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def replay(self, batch_size=32):
        with self.memory_lock:
            if len(self.memory) == 0:  # Kiểm tra nếu không có dữ liệu
                return
            current_batch_size = min(batch_size, len(self.memory))
            minibatch = random.sample(self.memory, current_batch_size)
        states = np.zeros((current_batch_size, self.board_size, self.board_size, 1))
        targets = np.zeros((current_batch_size, self.action_size))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            with self.model_lock:
                target = self.model.predict(state, verbose=0)[0]
                if done:
                    target[action] = reward
                else:
                    t = self.target_model.predict(next_state, verbose=0)[0]
                    target[action] = reward + self.gamma * np.amax(t)
            states[i] = state[0]
            targets[i] = target
        with self.model_lock:
            loss = self._tf_train_step(
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(targets, dtype=tf.float32)
            )
            # Lưu dữ liệu huấn luyện
            self.loss_history.append(loss.numpy())
            self.reward_history.append(np.mean([m[2] for m in minibatch]))  # Reward trung bình
            self.epsilon_history.append(self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()

    def load(self, name):
        with self.model_lock:
            self.model.load_weights(name)
            self.target_model.load_weights(name)

    def save(self, name):
        with self.model_lock:
            self.model.save(name)

    def set_mcts_params(self, simulations=None, c_puct=None, temperature=None):
        if not self.use_mcts:
            print("MCTS is not enabled")
            return
        if simulations is not None:
            self.mcts.num_simulations = simulations
        if c_puct is not None:
            self.mcts.c_puct = c_puct
        if temperature is not None:
            self.mcts.temperature = temperature

    def enable_mcts(self, enable=True, simulations=50):
        self.use_mcts = enable
        if enable and not hasattr(self, 'mcts'):
            self.mcts = MCTS(
                model=self.model,
                board_size=self.board_size,
                num_simulations=simulations,
                c_puct=2.0,
                temperature=1.0
            )
        elif enable:
            self.mcts.num_simulations = simulations

    def train_with_mcts(self, state, game, batch_size=32):
        if not self.use_mcts:
            return
        action, probabilities = self.mcts.search(state[0, :, :, 0], game)
        game_copy = copy.deepcopy(game)
        if action == self.board_size * self.board_size:
            game_copy.make_move('pass')
        else:
            x, y = divmod(action, self.board_size)
            game_copy.make_move((x, y))
        next_state = np.reshape(game_copy.get_state(), [1, self.board_size, self.board_size, 1])
        reward = game_copy.get_reward(game.current_player)
        done = game_copy.pass_count >= 2
        self.remember(state, action, reward, next_state, done)
        self.replay(batch_size)
        return action

    def plot_training_progress(self, bot_name=""):
        """Vẽ biểu đồ tiến trình huấn luyện sau khi hoàn tất."""
        plt.figure(figsize=(15, 5))

        # Biểu đồ Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()

        # Biểu đồ Reward
        plt.subplot(1, 3, 2)
        plt.plot(self.reward_history, label='Average Reward')
        plt.title('Average Reward per Batch')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()

        # Biểu đồ Epsilon
        plt.subplot(1, 3, 3)
        plt.plot(self.epsilon_history, label='Epsilon')
        plt.title('Epsilon Decay')
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.legend()

        # Lưu biểu đồ thành file và hiển thị
        plt.tight_layout()
        plt.savefig(f'training_progress_{bot_name}_{self.board_size}x{self.board_size}.png')
        plt.show()
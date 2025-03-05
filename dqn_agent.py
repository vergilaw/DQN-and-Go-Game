import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os


class DQNAgent:
    def __init__(self, board_size=19, model_path=None):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.update_target_frequency = 500
        self.step_counter = 0

        #GPU detection
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

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.board_size, self.board_size, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        # avoid exploding gradients
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("Target model updated")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)

        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            act_values = self.model.predict(state, verbose=0)

        return np.argmax(act_values[0])

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([item[0] for item in minibatch])
        next_states = np.vstack([item[3] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])

        # Double DQN implementation
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            q_values = self.model.predict(next_states, verbose=0)
            best_actions = np.argmax(q_values, axis=1)

            target_q_values = self.target_model.predict(next_states, verbose=0)

            targets = rewards + self.gamma * np.array([target_q_values[i, best_actions[i]] * (1 - dones[i])
                                                       for i in range(batch_size)])

            target_f = self.model.predict(states, verbose=0)
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]

            history = self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)

        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0] if 'loss' in history.history else None

    def save_model(self, path):
        self.model.save(path.replace('.h5', '.keras'))
        print(f"Model saved to {path.replace('.h5', '.keras')}")

    def load_model(self, path):
        try:
            self.model = load_model(path)
            self.target_model = load_model(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

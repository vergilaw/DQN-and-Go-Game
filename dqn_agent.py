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

        # GPU detection
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

    # Add this new method with the @tf.function decorator
    @tf.function(reduce_retracing=True)
    def predict_action(self, state):
        """Make predictions with the model using reduced retracing."""
        return self.model(state, training=False)

    # Add this new method with the @tf.function decorator
    @tf.function(reduce_retracing=True)
    def predict_target(self, state):
        """Make predictions with the target model using reduced retracing."""
        return self.target_model(state, training=False)

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Ensure consistent tensor shape
        if len(state.shape) == 2:  # If shape is (board_size, board_size)
            state = np.expand_dims(np.expand_dims(state, axis=0), axis=-1)
        elif len(state.shape) == 3:  # If shape is (board_size, board_size, 1)
            state = np.expand_dims(state, axis=0)

        # Convert to tensor
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            # Use the decorated function instead of model.predict
            act_values = self.predict_action(state_tensor)

        return np.argmax(act_values[0])

    # Add this method with the @tf.function decorator for training
    @tf.function(reduce_retracing=True)
    def train_step(self, states, target_f):
        """Perform a single training step with reduced retracing."""
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.keras.losses.MSE(target_f, predictions)
            # You can also use the following for a more direct approach:
            # loss = tf.reduce_mean(tf.square(target_f - predictions))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([item[0] for item in minibatch])
        next_states = np.vstack([item[3] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])

        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            # Use decorated functions instead of predict
            q_values = self.predict_action(next_states_tensor)
            best_actions = tf.argmax(q_values, axis=1).numpy()

            target_q_values = self.predict_target(next_states_tensor)

            # Calculate targets
            targets = rewards + self.gamma * np.array([target_q_values[i, best_actions[i]].numpy() * (1 - dones[i])
                                                       for i in range(batch_size)])

            # Get current predictions for all actions
            target_f = self.predict_action(states_tensor).numpy()

            # Update only the actions that were taken
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]

            # Convert target_f to tensor
            target_f_tensor = tf.convert_to_tensor(target_f, dtype=tf.float32)

            # Use the decorated training function
            loss = self.train_step(states_tensor, target_f_tensor)

        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.numpy() if hasattr(loss, 'numpy') else loss

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

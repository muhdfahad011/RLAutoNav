import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from car_environment import CarEnv

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
EPISODES = 1000

env = CarEnv()

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(120, 160, 1)))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard])

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

agent = DQNAgent()

for episode in range(EPISODES):
    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > 0.1:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 3)

        new_state, reward, done, _ = env.step(action)

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        episode_reward += reward
        step += 1

    print(f"Episode: {episode+1}, Reward: {episode_reward}, Steps: {step}")

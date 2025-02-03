import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

class LinearQNet(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')  
        self.linear2 = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.linear1(x)  # ReLU activation is included in the layer
        x = self.linear2(x)  # Linear output layer
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name) 
        self.save_weights(file_name)       

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        self.criterion = tf.keras.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int64)  
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        if len(state.shape)==1:
            state = tf.expand_dims(state, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            action = tf.expand_dims(action, axis=0)
            reward = tf.expand_dims(reward, axis=0)
            done = (done,)

        # Bellaman Equation

        # predicted Q value
        pred = self.model.predict(state)

        target = tf.identity(pred)  # Clone predictions
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state[idx], training=False))

            action_idx = tf.argmax(action[idx]).numpy()  # Get action index
            target = tf.tensor_scatter_nd_update(target, [[idx, action_idx]], [Q_new])

        # Q_new = r + y * max(next_predicted Q value) only if not done
        with tf.GradientTape() as tape:
            loss = self.criterion(target, pred)  # Compute loss

        gradients = tape.gradient(loss, self.model.trainable_variables)  # Compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  # Update weights

         




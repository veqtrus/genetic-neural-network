import random

import tensorflow as tf
from tensorflow import keras

import genetic_algorithm as ga

class GenAlgNN(ga.GAProblem):
    max_depth: int = 4
    max_width: int = 8
    epochs: int = 1000
    learning_rate: float = 0.03
    prob_cross: float = 0.9
    prob_mut: float = 0.05

    def __init__(self, training_set, validation_set):
        self.training_set = training_set
        self.validation_set = validation_set
    
    def train_nn(self, x):
        class PreprocessingLayer(keras.layers.Layer):
            def __init__(self):
                super(PreprocessingLayer, self).__init__()

            def call(self, inputs):
                return tf.concat([
                    inputs,
                    tf.square(inputs),
                    tf.sin(inputs)
                ], 1)
        
        model = keras.models.Sequential()
        model.add(PreprocessingLayer())
        for w in x:
            if w > 1:
                model.add(keras.layers.Dense(w, activation='tanh', kernel_regularizer=keras.regularizers.l2(l=0.01)))
        model.add(keras.layers.Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=self.learning_rate))
        X_train, y_train = self.training_set
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=8, verbose=0)
        return model

    def fitness(self, x):
        X_valid, y_valid = self.validation_set
        model = self.train_nn(x)
        y_pred = model.predict(X_valid)
        mse = keras.losses.MeanSquaredError()
        mse = mse(y_valid, y_pred).numpy()
        return 1.0 / (1.0 + mse)

    def generate_individual(self):
        return [random.randint(0, self.max_width) for _ in range(self.max_depth)]

    def crossover(self, a, b): # uniform crossover
        if random.random() < self.prob_cross:
            a = a.copy()
            b = b.copy()
            for i in range(self.max_depth):
                if random.random() < 0.5:
                    a[i], b[i] = b[i], a[i]
        return a, b

    def mutate(self, x):
        new_x = []
        for b in x:
            if random.random() < self.prob_mut:
                new_x.append(random.randint(0, self.max_width))
            else:
                new_x.append(b)
        return new_x

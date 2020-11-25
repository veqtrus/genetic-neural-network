import random

from tensorflow import keras

import genetic_algorithm as ga

class GenAlgNN(ga.GAProblem):
    hidden_layers: int = 8
    epochs: int = 400
    learning_rate: float = 0.03
    prob_cross: float = 1.0
    prob_mut: float = 0.1

    def __init__(self, training_set, validation_set):
        self.training_set = training_set
        self.validation_set = validation_set

    def fitness(self, x):
        layers = []
        for w in x:
            if w != 0: # can choose fewer layers
                layers.append(keras.layers.Dense(w, activation='tanh'))
        layers.append(keras.layers.Dense(1, activation='sigmoid'))
        model = keras.models.Sequential(layers)
        model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate))
        X_train, y_train = self.training_set
        X_valid, y_valid = self.validation_set
        model.fit(X_train, y_train, epochs=self.epochs, verbose=0)
        y_pred = model.predict(X_valid)
        mse = keras.losses.MeanSquaredError()
        mse = mse(y_valid, y_pred).numpy()
        return 1.0 / (1.0 + mse)

    def generate_individual(self):
        return [random.randint(0, 16) for _ in range(self.hidden_layers)]

    def crossover(self, a, b):
        if random.random() < self.prob_cross:
            cut = random.randrange(1, self.hidden_layers)
            new_a = []
            new_b = []
            for i in range(self.hidden_layers):
                if i < cut:
                    new_a.append(a[i])
                    new_b.append(b[i])
                else:
                    new_a.append(b[i])
                    new_b.append(a[i])
            return new_a, new_b
        return a, b

    def mutate(self, x):
        new_x = []
        for i in x:
            if random.random() < self.prob_mut:
                r = i
                while r == i:
                    r = random.randint(0, 16)
                new_x.append(r)
            else:
                new_x.append(i)
        return new_x

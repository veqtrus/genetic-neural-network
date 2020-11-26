import random

import tensorflow as tf
from tensorflow import keras

import genetic_algorithm as ga

class GPNode:
    links: list = []

    def get_links(self) -> list:
        res = self.links.copy()
        for link in self.links:
            if link is None:
                continue
            if link.node is None:
                continue
            res.extend(link.node.get_links())
        return res

    def build(self):
        pass
    
    def copy(self):
        pass
    
    def to_dot(self) -> str:
        label = 'ID' + str(id(self))
        lines = [label + ' [label="' + str(self) + '"];']
        for link in self.links:
            if link is not None and link.node is not None:
                lines.append('ID' + str(id(link.node)) + ' -> ' + label + ';')
                lines.append(link.node.to_dot())
        return '\n'.join(lines)

class GPLink:
    node: GPNode = None

    def copy(self):
        res = GPLink()
        res.node = self.node.copy()
        return res

class Layer(GPNode):
    width: int = 4

    def __init__(self, width: int):
        self.width = width
        self.links = [GPLink()]
    
    def build(self):
        input0 = self.links[0].node.build()
        return keras.layers.Dense(self.width, activation='tanh')(input0)
    
    def copy(self):
        res = Layer(self.width)
        res.links[0] = self.links[0].copy()
        return res
    
    def __str__(self):
        return 'Layer(' + str(self.width) + ')'

class Input(GPNode):
    def __init__(self, problem):
        self.problem = problem
        self.links = []
    
    def build(self):
        return self.problem.cur_input
    
    def copy(self):
        return Input(self.problem)
    
    def __str__(self):
        return 'Input'

class Concatenation(GPNode):
    def __init__(self):
        self.links = [GPLink(), GPLink()]

    def build(self):
        return keras.layers.concatenate([link.node.build() for link in self.links])
    
    def copy(self):
        res = Concatenation()
        res.links = []
        for i in range(len(self.links)):
            res.links.append(self.links[i].copy())
        return res
    
    def __str__(self):
        return 'Concatenation'

class Function(GPNode):
    fn = lambda x: x

    def __init__(self):
        self.links = [GPLink()]
    
    def build(self):
        class CustomLayer(keras.layers.Layer):
            def __init__(self, fn):
                super(CustomLayer, self).__init__()
                self.fn = fn

            def call(self, inputs):
                return self.fn(inputs)
        return CustomLayer(self.fn)(self.links[0].node.build())
    
    def copy(self):
        res = Function()
        res.fn = self.fn
        res.links[0] = self.links[0].copy()
        return res
    
    def __str__(self):
        if self.fn == tf.square:
            s = 'square'
        elif self.fn == tf.sin:
            s = 'sin'
        else:
            s = str(self.fn)
        return 'Function(' + s + ')'

class GenProgNN(ga.GAProblem):
    max_width: int = 8
    max_depth: int = 4
    epochs: int = 1000
    learning_rate: float = 0.03
    prob_cross: float = 0.9
    prob_mut: float = 0.05

    cur_input = None

    node_choices = [Layer, Input, Concatenation, Function]

    def __init__(self, training_set, validation_set):
        self.training_set = training_set
        self.validation_set = validation_set
    
    def train_nn(self, x):
        self.cur_input = keras.layers.Input(shape=[2])
        model = keras.Model(inputs=[self.cur_input], outputs=[x.build()])
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

    def make_substree(self, depth):
        if depth > self.max_depth:
            node_type = Input
        else:
            node_type = random.choice(self.node_choices)
        node: GPNode = None
        if node_type is Layer:
            node = Layer(random.randint(1, self.max_width))
        elif node_type is Input:
            node = Input(self)
        elif node_type is Function:
            node = Function()
            node.fn = random.choice([tf.square, tf.sin])
        else:
            node = node_type()
        for i in range(len(node.links)):
            node.links[i].node = self.make_substree(depth + 1)
        return node

    def generate_individual(self):
        node = Layer(1)
        node.links[0].node = self.make_substree(1)
        return node

    def crossover(self, a, b):
        if random.random() < self.prob_cross:
            a = a.copy()
            b = b.copy()
            link_a = random.choice(a.get_links())
            link_b = random.choice(b.get_links())
            link_a.node, link_b.node = link_b.node, link_a.node
        return a, b

    def mutate(self, x):
        x = x.copy()
        for link in x.get_links():
            node: GPNode = link.node
            if random.random() < self.prob_mut:
                if isinstance(node, Layer):
                    node.width = random.randint(1, self.max_width)
                elif isinstance(node, Function):
                    if node.fn == tf.square:
                        node.fn = tf.sin
                    elif node.fn == tf.sin:
                        node.fn = tf.square
        return x

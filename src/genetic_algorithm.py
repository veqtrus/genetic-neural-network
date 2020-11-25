import random

class GAProblem:
    def fitness(self, x):
        pass

    def generate_individual(self):
        pass

    def crossover(self, a, b):
        pass

    def mutate(self, x):
        pass

class GAOptimizer:
    problem: GAProblem = None
    population: list = []

    def __init__(self, problem: GAProblem):
        self.problem = problem

    def reset(self):
        self.population = []
    
    def start(self, pop_size: int):
        self.reset()
        for _ in range(pop_size):
            self.population.append(self.problem.generate_individual())
    
    def next_generation(self):
        fitnesses = [self.problem.fitness(i) for i in self.population]
        def selection():
            return random.choices(self.population, weights=fitnesses)[0]
        new_population = []
        while len(new_population) < len(self.population):
            child_a, child_b = self.problem.crossover(selection(), selection())
            child_a = self.problem.mutate(child_a)
            child_b = self.problem.mutate(child_b)
            new_population.append(child_a)
            new_population.append(child_b)
        self.population = new_population
        return new_population, fitnesses

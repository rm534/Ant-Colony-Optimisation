import Data
import random

def distance(weight, value):
    return

class AntOptimiser:
    def __init__(self, data):               #TODO add to inputs for class
        self.data = data
        self._init_pheromone()
        self._init_ants_pop(10)
        return

    def _init_pheromone(self):
        treasure_num = 1
        while treasure_num <= 100:
            self.data["treasure"+str(treasure_num)]["pheromone"] = random.uniform(0.0, 1.0)
            treasure_num +=1

    def _init_ants_pop(self, population):
        self.ants = {}
        ants = 1
        path = []

        while ants <= population:
            init_node = random.randrange(1, 100, 1)
            path = []
            path.append(self.data["treasure"+str(init_node)])
            self.ants["ant"+str(ants)] = path
            ants += 1
        print(self.ants)

    def path_options(self, path):
        #TODO implement a function to display node options to pick_path with pheromone values
        for elements in path:

        return

    def pick_path(self):
        #TODO implement a path choice function
        return

    def create_path(self):
        #TODO implement a complete path creation based on number of ants
        return

    def update_pheromone(self):
        #TODO implement a pheromone update function based on fitness of solutions
        return
    def evaporate_pheromone(self):
        #TODO implement a evaporation function for the pheromone values
        return
    def evaluate_fitness(self):
        #TODO implement fitness function
        return
    def run_optimisation(self):
        #TODO implement the running of the optimisation algorithm
        return


if __name__ == "__main__":
    optimiser = AntOptimiser(Data.data)



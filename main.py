import Data
import random
from numpy.random import choice


def distance(weight, value):
    return


class AntOptimiser:
    def __init__(self, data):  # TODO add to inputs for class
        self.data = data
        self.population = 10
        self.max_tour_weight = 1000
        self.solution = []
        self.loop_done = False
        self._init_pheromone()
        self._init_ants_pop(self.population)
        return

    def _init_pheromone(self):
        treasure_num = 1
        while treasure_num <= 100:
            self.data["treasure" + str(treasure_num)]["pheromone"] = random.uniform(0.0, 1.0)
            treasure_num += 1

    def _init_ants_pop(self, population):
        self.ants = {}
        for x in range(1, population + 1):
            self.ants.update({"ant" + str(x):
                                  {"path": [],
                                   "choices": [],
                                   "prob": []}
                              })
        ants = 1

        while ants <= population:
            init_node = random.randrange(1, 100, 1)
            path = []
            choices = []
            for x in range(1, 101):
                choices.append(x)
            choices.remove(init_node)
            path.append(self.data["treasure" + str(init_node)]["ID"])
            self.ants["ant" + str(ants)]["path"] = path
            self.ants["ant" + str(ants)]["choices"] = choices
            ants += 1

    def path_choices(self, ant):
        for elements in self.ants[ant]["path"]:
            if elements in self.ants[ant]["choices"]:
                self.ants[ant]["choices"].remove(elements)

    def update_path_choices(self):
        for x in range(1, self.population + 1):
            self.path_choices("ant" + str(x))

    def path_path(self, ant):
        choices = self.ants[ant]["choices"]
        pheromone = []
        sum_pheromone = 0
        prob = []
        for elements in choices:
            pheromone.append(self.data["treasure" + str(elements)]["pheromone"])
            sum_pheromone += self.data["treasure" + str(elements)]["pheromone"]

        for elements in pheromone:
            _prob = elements / sum_pheromone
            prob.append(_prob)
        pick = choice(a=choices, size=1, p=prob)
        print(pick)
        self.ants[ant]["path"].append(int(pick[0]))
        # print(self.ants[ant]["path"])
        return

    def update_path_path(self):
        finished = 0
        for x in range(1, self.population + 1):
            if self.tour_weight("ant" + str(x)) > self.max_tour_weight:
                continue
            else:
                finished += 1
                self.path_path("ant" + str(x))
        if finished == 0:
            self.loop_done = True
        else:
            self.loop_done = False

    def tour_weight(self, ant):
        path = self.ants[ant]["path"]
        weights = []
        sum_weight = 0
        for node in path:
            weights.append(self.data["treasure" + str(node)]["weight"])
            sum_weight += int(self.data["treasure" + str(node)]["weight"])
        # print(sum_weight)
        return sum_weight

    def create_path(self):
        # TODO test and verify the path and the random choice
        while self.loop_done == False:
            self.update_path_path()
            self.update_path_choices()
        print(self.ants["ant1"]["path"])
        return

    def update_pheromone(self):
        # TODO implement a pheromone update function based on fitness of solutions
        return

    def evaporate_pheromone(self):
        # TODO implement a evaporation function for the pheromone values
        return

    def evaluate_fitness(self):
        # TODO implement fitness function
        for x in range(1, self.population + 1):
            solution = self.ants["ant" + str(x)]["path"]

        return

    def run_optimisation(self):
        # TODO implement the running of the optimisation algorithm
        return

    def access_dict(self, node, *argv):

if __name__ == "__main__":
    optimiser = AntOptimiser(Data.data)
    # print(optimiser.data)
    optimiser.create_path()

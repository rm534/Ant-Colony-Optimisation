import Data
import random
from numpy.random import choice
import numpy as np


def distance(weight, value):
    return


class AntOptimiser:
    def __init__(self, data, evaporation, pheromone_weight, population):  # TODO add to inputs for class
        self.data = data
        self.population = population
        self.max_tour_weight = 1000
        self.evaporation = evaporation
        self.pheromone_weight = pheromone_weight
        self.solution = []
        self.loop_done = False
        self.pheromones = [[0 for y in range(0, 101)] for x in range(0, 101)]
        self._init_pheromone()
        self._init_ants_pop()

        return

    def _init_pheromone(self):

        for i in range(1, 100):
            for j in range(1, 100):
                if i == j:
                    self.pheromones[i][j] = 0
                else:
                    self.pheromones[i][j] = random.uniform(0.0, 1.0)

    def _init_ants_pop(self):
        self.ants = {}
        for x in range(1, self.population + 1):
            self.ants.update({"ant" + str(x):
                                  {"path": [],
                                   "choices": [],
                                   "prob": []}
                              })
        ants = 1
        while ants <= self.population:
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

    def _reset_path(self):
        for x in range(1, self.population + 1):
            path = self.ants["ant" + str(x)]["path"]
            start = path[0]
            self.ants["ant" + str(x)]["path"] = []
            self.ants["ant" + str(x)]["path"].append(start)

        for x in range(1, self.population + 1):
            choices = []
            self.ants["ant" + str(x)]["choices"] = choices
            for y in range(1, 101):
                self.ants["ant" + str(x)]["choices"].append(y)

        self.loop_done = False
        return

    def path_choices(self, ant):
        for elements in self.ants[ant]["path"]:
            if elements in self.ants[ant]["choices"]:
                self.ants[ant]["choices"].remove(elements)

    def update_path_choices(self):
        for x in range(1, self.population + 1):
            self.path_choices("ant" + str(x))

    def path_path(self, ant):
        choices = self.ants[ant]["choices"]
        current_path = self.ants[ant]["path"]
        current_node = current_path[-1]

        pheromone = []
        sum_pheromone = 0
        prob = []
        # print(np.shape(choices), np.shape(self.pheromones))
        for elements in choices:
            pheromone.append(self.pheromones[current_node][elements])
            sum_pheromone += self.pheromones[current_node][elements]
        for elements in pheromone:
            _prob = elements / sum_pheromone
            prob.append(_prob)
        pick = choice(a=choices, size=1, p=prob)
        self.ants[ant]["path"].append(int(pick[0]))
        if not self.weight_check(ant):
            self.ants[ant]["path"].remove(int(pick[0]))
            return "finished"

        # print(self.ants[ant]["path"])
        # print(self.ants[ant]["choices"])
        # print("pick =", pick)
        # print(self.pheromones[1][2])

        return "TBA"

    def weight_check(self, ant):
        if self.tour_weight(ant) > self.max_tour_weight:
            return False
        else:
            return True

    def update_path_path(self):
        finished = 0
        for x in range(1, self.population + 1):
            # print("ant" + str(x) + "before path chosen:", self.tour_weight("ant" + str(x)))
            if self.path_path("ant" + str(x)) == "finished":
                continue
            else:
                finished += 1

            # print("ant" + str(x) + "after path chosen:", self.tour_weight("ant" + str(x)))
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

        return


    def update_fitness(self):
        wealth = []
        for x in range(1, self.population + 1):
            wealth.append(self.evaluate_fitness("ant" + str(x)))
        index = wealth.index(max(wealth))
        print("best solution wealth:",wealth[index])
        return "ant" + str(index + 1)

    def evaluate_fitness(self, ant):
        # TODO implement fitness function
        tour_value = self.tour_value(ant)
        wealth = float(tour_value)
        # print wealth
        return wealth

    def run_optimisation(self):
        # TODO implement the running of the optimisation algorithm
        return

    def access_dict(self, node, *argv):
        return

    def tour_value(self, ant):
        sum_value = 0
        path = self.ants[ant]["path"]
        for element in path:
            sum_value += float(self.data["treasure" + str(element)]["value"])

        return sum_value

    def evaporate_pheromone(self):

        for i in range(1, 100):
            for j in range(1, 100):
                if i == j:
                    self.pheromones[i][j] = 0
                else:
                    self.pheromones[i][j] = self.evaporation * self.pheromones[i][j]


    def deposit_pheromone(self, ant):
        path = self.ants[ant]["path"]
        for x in range(0, len(path) - 1):
            self.pheromones[path[x]][path[x + 1]] = self.pheromones[path[x]][path[x + 1]] + self.tour_value(
                ant) * self.pheromone_weight


    def update_pheromone(self):
        for x in range(1, self.population + 1):
            self.deposit_pheromone("ant" + str(x))
        self.evaporate_pheromone()

        # print(self.pheromones[1][2])


if __name__ == "__main__":
    optimiser = AntOptimiser(Data.data, evaporation=0.5, pheromone_weight=0.000000001, population=100)
    x = 0

    while x <= 10000:
        x += 1
        optimiser.create_path()
        print(optimiser.update_fitness())
        optimiser.update_pheromone()
        optimiser._reset_path()

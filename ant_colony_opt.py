import Data
import random
from numpy.random import choice
import numpy as np


# import plotting_tools as pt

def distance(weight, value):
    return


MAX_TOUR_WEIGHT = 1000
FITNESS_EVALUATIONS = 10000


class AntOptimiser:
    def __init__(self, data, evaporation, pheromone_weight, population):
        self.data = data
        self.evaporation = evaporation
        self.pheromone_weight = pheromone_weight
        self.population = population

        self.loop_done = False
        self.max_value = self._init_max_value()
        self._init_pheromone()
        self._init_ants_pop()

        return

    def _init_max_value(self):
        max_value = 0
        for i in range(1, 101):
            max_value += float(self.data["treasure" + str(i)]["value"])

        return max_value

    def _init_pheromone(self):
        b = np.random.uniform(0.0, 1.0, size=(101,101))
        self.pheromones = (b+b.T)/2
        for i in range(0, 100):
            for j in range(0, 100):
                if i == j:
                    self.pheromones[i][j] = 0

        print(np.shape(self.pheromones))
        #print(self.pheromones[0])

    def _init_ants_pop(self):
        # TODO add start node and end node
        self.ants = {}
        for x in range(1, self.population + 1):
            self.ants.update({"ant" + str(x):
                                  {"path": [],
                                   "choices": [],
                                   "prob": []}
                              })
        ants = 1
        while ants <= self.population:
            path = []
            choices = []
            for x in range(1, 101):
                choices.append(x)
            init_node = self.choose_path("ant"+str(ants),choices,current_node=0)
            choices.remove(init_node)
            path.append(self.data["treasure" + str(init_node[0])]["ID"])
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

    def choose_path(self, ant, choices, current_node):
        pheromone = []
        sum_pheromone = 0
        prob = []

        for elements in choices:
            pheromone.append(self.pheromones[current_node][elements])
            sum_pheromone += self.pheromones[current_node][elements]
        for elements in pheromone:

            _prob = elements / sum_pheromone
            prob.append(_prob)
        pick = choice(a=choices, size=1, p=prob)
        self.ants[ant]["path"].append(int(pick[0]))

        return pick

    def path_path(self, ant):
        choices = self.ants[ant]["choices"]
        current_path = self.ants[ant]["path"]
        #print(current_path)
        current_node = current_path[-1]
        pick = self.choose_path(ant, choices, current_node)
        if not self.weight_check(ant):
            self.ants[ant]["path"].remove(int(pick[0]))
            return "finished"

        # print(self.ants[ant]["path"])
        # print(self.ants[ant]["choices"])
        # print("pick =", pick)
        # print(self.pheromones[1][2])

        return "TBA"

    def weight_check(self, ant):
        if self.tour_weight(ant) > MAX_TOUR_WEIGHT:
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

        while self.loop_done == False:
            self.update_path_path()
            self.update_path_choices()

        return

    def update_fitness(self):
        wealth = []
        for x in range(1, self.population + 1):
            wealth.append(self.evaluate_fitness("ant" + str(x)))
        index = wealth.index(max(wealth))
        print("best solution wealth:", wealth[index])
        return "ant" + str(index + 1), wealth[index]

    def evaluate_fitness(self, ant):
        tour_value = self.tour_value(ant)
        wealth = float(tour_value)
        # print wealth
        return wealth

    def run_optimisation(self):
        solution = []
        for i in range(0, FITNESS_EVALUATIONS):
            self.create_path()
            solution_single = self.update_fitness()
            solution.append(solution_single[1])
            print(solution_single[0])
            self.update_pheromone()
            self._reset_path()

        self.write_solution(solution)

        return

    def write_solution(self, solution):
        name = "population:" + str(self.population) + " " + "evaporation:" + str(
            self.evaporation) + " " + "pheromone value:" + str(self.pheromone_weight) + ".txt"
        f = open(name, "w+")
        for i in range(0, FITNESS_EVALUATIONS):
            f.write("solution" + str(i) + ":" + str(solution[i]) + "\n")
        f.close()
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
                    self.pheromones[i][j] = (1 - self.evaporation) * self.pheromones[i][j]

    def deposit_pheromone(self, ant):
        path = self.ants[ant]["path"]
        for x in range(0, len(path) - 1):
            self.pheromones[path[x]][path[x + 1]] += self.pheromone_weight * (self.tour_value(ant) / self.max_value)

    def update_pheromone(self):
        self.evaporate_pheromone()
        for x in range(1, self.population + 1):
            self.deposit_pheromone("ant" + str(x))
        print("pheromone =", self.pheromones[22][55])


        # print(self.pheromones[1][2])


if __name__ == "__main__":
    optimiser = AntOptimiser(Data.data, evaporation=0.5, pheromone_weight=1000000000000000, population=40)
    optimiser.run_optimisation()

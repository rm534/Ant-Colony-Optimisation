import Data
from numpy.random import choice
import numpy as np
import time

# Weight Restrictions and Fitness Evaluation Constants Definition
MAX_TOUR_WEIGHT = 1000
FITNESS_EVALUATIONS = 10000


# Ant Optimiser Object
class AntOptimiser:
    def __init__(self, data, evaporation, pheromone_weight, population, pheromone_min, pheromonone_max):
        self.time_to_finish = 0
        self.data = data
        self.evaporation = evaporation
        self.pheromone_weight = pheromone_weight
        self.population = population
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromonone_max
        self.loop_done = False
        self.max_value = self._init_max_value()
        self._init_pheromone()
        self._init_ants_pop()
        self.start = time.time()
        self.loop_time = self._init_loop_time()

        return

    # Function to time one loop of the optimisation and from this the time remaining will be estimated
    def _init_loop_time(self):
        start = time.time()
        self.create_path()
        self.update_fitness()
        self.update_pheromone()
        self._reset_path()
        end = time.time() - start
        return end

    def _init_max_value(self):
        max_value = 0
        for i in range(1, 101):
            max_value += float(self.data["treasure" + str(i)]["value"])

        return max_value

    # Function to initialise the pheromone matrix, creating an extra column and
    def _init_pheromone(self):
        b = np.random.uniform(0.0, 1.0, size=(
        101, 101))  # creating new matrix with inital pheromone values 0-1 to hold pheromones
        self.pheromones = (b + b.T) / 2  # transforming the matrix to a symetrical matrix about the diagonal
        for i in range(0, 100):  # initialising the diagonal as zero
            self.pheromones[i][i] = 0

    # Function to initialise the population of ants
    def _init_ants_pop(self):
        self.ants = {}  # creating a dictionary to hold ants
        for x in range(1, self.population + 1):  # initialising ants for number of ants required
            self.ants.update({"ant" + str(x):  # initialising each ant in population with var path, choices
                                  {"path": [],
                                   "choices": []}
                              })
        ants = 1
        while ants <= self.population:
            path = []
            choices = []
            for x in range(1, 101):  # initialising initial node and choices array
                choices.append(x)
            init_node = self.choose_path("ant" + str(ants), choices, current_node=0)
            choices.remove(init_node)
            path.append(self.data["treasure" + str(init_node[0])][
                            "ID"])  # adding initial node from treasure data (ID returns the number of this treasure
            self.ants["ant" + str(ants)]["path"] = path
            self.ants["ant" + str(ants)]["choices"] = choices
            ants += 1

    # Function to reset path after a successful evaluation
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
                if y == start:
                    continue
                self.ants["ant" + str(x)]["choices"].append(y)
        self.loop_done = False
        return

    # Function to calculate the time left of the optimisation
    def time_left(self, evaluation, time_taken):
        time_left = (FITNESS_EVALUATIONS - evaluation) * time_taken
        time_left /= 60

        return int(time_left)

    # Function to evaluate if elements in path are also in choices, if so then choices removes this element
    def path_choices(self, ant):
        for elements in self.ants[ant]["path"]:
            if elements in self.ants[ant]["choices"]:
                self.ants[ant]["choices"].remove(elements)

    # Function to update choices array for each ant
    def update_path_choices(self):
        for x in range(1, self.population + 1):
            self.path_choices("ant" + str(x))

    # Function to choose path based on pheromone graph and choices available
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

    # Function to evaluate a path element of an ant
    def path_path(self, ant):
        choices = self.ants[ant]["choices"]
        current_path = self.ants[ant]["path"]
        current_node = current_path[-1]
        pick = self.choose_path(ant, choices, current_node)
        if not self.weight_check(ant):
            self.ants[ant]["path"].remove(pick)
            return "finished"
        else:
            return "TBA"

    # Function to evaluate the weight of a path and return if it exceeds the weight limit
    def weight_check(self, ant):
        if self.tour_weight(ant) > MAX_TOUR_WEIGHT:
            return False
        else:
            return True

    # Function to update path for each ant until all ants have a complete path that does not violate the restriction
    def update_path_path(self):
        finished = 0

        for x in range(1, self.population + 1):
            if self.path_path("ant" + str(x)) == "finished":
                continue
            else:
                finished += 1
        if finished == 0:
            self.loop_done = True
        else:
            self.loop_done = False

    # Function to calculate the sum of weights in a path
    def tour_weight(self, ant):
        path = self.ants[ant]["path"]
        weights = []
        sum_weight = 0
        for node in path:
            weights.append(self.data["treasure" + str(node)]["weight"])
            sum_weight += int(self.data["treasure" + str(node)]["weight"])
        return sum_weight

    # Function to create a path for all ants by calling update_path_path and update_path_choices
    def create_path(self):
        while self.loop_done == False:
            self.update_path_path()
            self.update_path_choices()

        return

    # Function to update the fitness of all ants paths returning the best ant, best wealth, average wealth and an array of all wealth values
    def update_fitness(self):
        wealth = []
        wealth_sum = 0
        for x in range(1, self.population + 1):
            wealth.append(self.tour_value("ant" + str(x)))
            wealth_sum += self.tour_value("ant" + str(x))
        wealth_avg = wealth_sum / len(wealth)
        index = wealth.index(max(wealth))
        print("best solution wealth:", wealth[index])
        print("best solution weight:", self.tour_weight("ant" + str(index + 1)))
        print("avg solution wealth:", wealth_avg)
        print("time left:", self.time_to_finish, "mins")
        return "ant" + str(index + 1), wealth[index], wealth_avg, wealth

    # Function to run the optimisation for a full cycle of fitness evaluations, as well as appending the relevant data and writing it to an output after all evaluations are finished
    # after all evaluations are finished
    def run_optimisation(self):
        solution = []
        solutions = []
        average = []
        for i in range(0, FITNESS_EVALUATIONS):
            self.create_path()
            solutionAll = self.update_fitness()
            print(solutionAll[0])
            self.update_pheromone()
            solution.append(solutionAll[1])
            average.append(solutionAll[2])
            solutions.append(solutionAll[3])
            self._reset_path()

            self.time_to_finish = self.time_left(i, self.loop_time)

        self.write_solution(solution, average, solutions)

        return

    # Function to write solutions to three seperate files
    def write_solution(self, solution, average_solution, solutions):
        name = "population:" + str(self.population) + " " + "evaporation:" + str(
            self.evaporation) + " " + "pheromone value:" + str(self.pheromone_weight) + ".txt"
        f = open(name, "w+")
        for i in range(0, FITNESS_EVALUATIONS):
            f.write("solution" + str(i) + ":" + str(solution[i]) + "\n")
        f.close()
        name_avg = "population:" + str(self.population) + " " + "evaporation:" + str(
            self.evaporation) + " " + "pheromone value:" + str(self.pheromone_weight) + "average solution" + ".txt"
        f = open(name_avg, "w+")
        for i in range(0, FITNESS_EVALUATIONS):
            f.write("average" + str(i) + ":" + str(average_solution[i]) + "\n")
        f.close()
        name_solutions = "population:" + str(self.population) + " " + "evaporation:" + str(
            self.evaporation) + " " + "pheromone value:" + str(self.pheromone_weight) + "all solutions" + ".txt"
        f = open(name_solutions, "w+")
        for i in range(0, FITNESS_EVALUATIONS):
            f.write("solutions" + str(i) + ":" + str(solutions[i]) + "\n")
        f.close()
        return

    # Function to sum wealth of a path
    def tour_value(self, ant):
        sum_value = 0
        path = self.ants[ant]["path"]
        for element in path:
            sum_value += float(self.data["treasure" + str(element)]["value"])

        return sum_value

    # Function to evaporate all the pheromones
    def evaporate_pheromone(self):

        for i in range(1, 100):
            for j in range(1, 100):
                if i == j:
                    self.pheromones[i][j] = 0
                else:
                    self.pheromones[i][j] = (1 - self.evaporation) * self.pheromones[i][j]

    # Function to deposit pheromones with fitness function
    def _deposit_pheromone(self, ant):
        path = self.ants[ant]["path"]
        for x in range(0, len(path) - 1):
            self.pheromones[path[x]][path[x + 1]] += self.pheromone_weight * (
                    self.tour_value(ant) / self.tour_weight(ant))

    # Function to deposit pheromones for all ants
    def deposit_pheromone(self):
        for x in range(1, self.population + 1):
            self._deposit_pheromone("ant" + str(x))

    # Function to limit max and min pheromone values
    def check_min_max_pheromone(self):
        for i in range(1, 100):
            for j in range(1, 100):
                if self.pheromones[i][j] <= self.pheromone_min:
                    self.pheromones[i][j] = self.pheromone_min
                elif self.pheromones[i][j] >= self.pheromone_max:
                    self.pheromones[i][j] = self.pheromone_max
                elif self.pheromones[i][j] > 6:
                    print(self.pheromones[i][j])

    # Function to fully update pheromone values
    def update_pheromone(self):
        self.evaporate_pheromone()
        self.deposit_pheromone()
        self.check_min_max_pheromone()


if __name__ == "__main__":
    optimiser = AntOptimiser(Data.data, evaporation=0.05, pheromone_weight=1, population=50, pheromone_min=0.01,
                             pheromonone_max=10)
    optimiser.run_optimisation()

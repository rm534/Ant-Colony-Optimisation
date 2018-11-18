import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_solution(population, evaporation, pheromone_weight):
    solutions_y = []
    count_x = []
    name = "population:" + str(population) + " " + "evaporation:" + str(
        evaporation) + " " + "pheromone value:" + str(pheromone_weight) + ".txt"
    file = open(name, 'r')
    file_data = file.readlines()
    # print(file_data)

    for i in range(0, len(file_data)):

        try:
             solution = extract_solution(file_data[i])
             solutions_y.append(float(solution))

        except ValueError:
            pass


    for i in range(0, len(file_data)):
        count_x.append(i+1)
    print(np.shape(solutions_y), np.shape(count_x))
    return count_x, solutions_y


def extract_solution(string):
    solution = 0
    new = string.split(":")

    for element in new:
        if '\n' in element:
            # element.replace('\n', "")
            # element.replace('.0', "")
            solution = element
    solution = solution.replace("\n", "")
    # solution = solution.replace(" ", "")

    return solution


def plot_values_update(count_x, solution_y):
    fig, ax = plt.subplots()
    ax.plot(count_x, solution_y)
    ax.set(xlabel='time(s)', ylabel='voltage (mV)',
           title='Fitness over evaluations')
    ax.grid()
    fig.savefig("test.png")
    plt.show()

    return


#
#
# def plot_evol_best_solution(y_solution, x_count):
#     try:
#         pw = pg.plot(x_count, y_solution, pen='r')
#     except err as error:
#         print("plot failed")
#     return True


if __name__ == "__main__":
    count_x, solutions_y = read_solution(population=100, evaporation=0.05, pheromone_weight=0.8)
    plot_values_update(count_x, solutions_y)

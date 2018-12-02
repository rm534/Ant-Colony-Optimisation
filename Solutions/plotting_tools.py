import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_solution(population, evaporation, pheromone_weight, data_type):
    solutions_y = []
    solutions_x = []
    if data_type == 'best':
        name = "population:" + str(population) + " " + "evaporation:" + str(
            evaporation) + " " + "pheromone value:" + str(pheromone_weight) + ".txt"

    elif data_type == 'average':
        name = "population:" + str(population) + " " + "evaporation:" \
               + str(evaporation) + " " + "pheromone value:" + str(pheromone_weight) + "average solution" + ".txt"

    elif data_type == "all ants":
        name = "population:" + str(population) + " " + "evaporation:" \
               + str(evaporation) + " " + "pheromone value:" + str(pheromone_weight) + "all solutions" + ".txt"
    else:
        name = "population:" + str(population) + " " + "evaporation:" + str(
            evaporation) + " " + "pheromone value:" + str(pheromone_weight) + ".txt"
    file_data = open_file(name)
    solutions_y = extract_solution(file_data)

    for x in range(0, len(solutions_y)):
        solutions_x.append(x)
    return solutions_x, solutions_y


def open_file(name):
    file = open(name, 'r')
    file_data = file.readlines()
    return file_data


def extract_solution(lines):
    solution = 0
    solutions = []
    if 'solution_all' in lines[0]:
        solution = []
        for elements in lines:
            new = elements.split(":")
            for element in new:
                if '[' in element:
                    part = element
                else:
                    pass
            part = part.replace("[", "")
            part = part.replace("]", "")
            part = part.replace('\n', "")
            part = part.split(",")

            for parts in part:
                solution.append(float(parts))

            solutions.append(solution)
        return solutions

    elif 'solution' in lines[0]:
        print("gello")
        for elements in lines:
            new = elements.split(":")
            for element in new:
                if '\n' in element:
                    # element.replace('\n', "")
                    # element.replace('.0', "")
                    solution = element
            solution = solution.replace("\n", "")
            solutions.append(float(solution))
        return solutions

    elif 'average' in lines[0]:
        for elements in lines:
            new = elements.split(":")
            for element in new:
                if '\n' in element:
                    # element.replace('\n', "")
                    # element.replace('.0', "")
                    solution = element
            solution = solution.replace("\n", "")
            solutions.append(float(solution))
        return solutions


def plot(x, y, index, title_x, title_y, title):

    plt.subplot(1, 1, index)
    plt.scatter(x, y, marker=',', s=0.5, c='r')
    plt.ylabel(title_y+'\n'+'Fitness')
    if index == 1:
        plt.title(title)
    # elif index == 4:
    plt.xlabel(title_x)


def plot_values_update(solution_x, solution_y):

    x1, y1 = read_solution(10, 0.05, 0.03, 'best')
    x2, y2 = read_solution(20, 0.05, 0.03, 'best')
    x3, y3 = read_solution(50, 0.05, 0.03, 'best')
    x4, y4 = read_solution(100, 0.05, 0.03, 'best')
    x5, y5 = read_solution(50, 0.05, 0.03, 'best')
    x6, y6 = read_solution(50, 0.05, 0.1, 'best')
    x7, y7 = read_solution(50, 0.05, 0.5, 'best')
    x8, y8 = read_solution(50, 0.05, 1, 'best')
    x9, y9 = read_solution(50, 0.05, 0.03, 'best')
    x10, y10 = read_solution(50, 0.1, 0.03, 'best')
    x11, y11 = read_solution(50, 0.2, 0.03, 'best')
    x12, y12 = read_solution(50, 0.5, 0.03, 'best')
    x13, y13 = read_solution(100, 0.05, 0.03, 'average')

    plt.title('Varying Pheromone')
    # plot(x1, y1, 1, 'Evaluations','Population: 10', 'Varying Population')
    # plot(x2, y2, 2, 'Evaluations','Population: 20', 'Varying Population')
    # plot(x3, y3, 3, 'Evaluations','Population: 50', 'Varying Population')
    #plot(x4, y4, 1, 'Evaluations','Population: 100', 'Best Solution')
    # plot(x5, y5, 1, 'Evaluations','Pheromone: 0.03', 'Varying Pheromone')
    # plot(x6, y6, 2, 'Evaluations','Pheromone: 0.1', 'Varying Pheromone')
    # plot(x7, y7, 3, 'Evaluations','Pheromone: 0.5', 'Varying Pheromone')
    # plot(x8, y8, 4, 'Evaluations','Pheromone: 1', 'Varying Pheromone')
    # plot(x9, y9, 1, 'Evaluations','Evaporation: 0.05', 'Varying Evaporation')
    # plot(x10, y10, 2, 'Evaluations','Evaporation: 0.1', 'Varying Evaporation')
    # plot(x11, y11, 3, 'Evaluations','Evaporation: 0.2', 'Varying Evaporation')
    # plot(x12, y12, 4, 'Evaluations','Evaporation: 0.5', 'Varying Evaporation')
    plot(x13, y13, 1, 'Evaluations', 'Population: 100', 'Best Solution')
    plt.savefig('figure1.png')
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
    solutions_x, solutions_y = read_solution(20, 0.05, 0.03, 'best')
    plot_values_update(solutions_x, solutions_y)

import Topology
import RWA
import Optical_Network
import networkx as nx
import logging
import pandas as pd
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
ON = Optical_Network.OpticalNetwork("nsf")


def time_func(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        diff = end - start
        return diff

    return wrapper


def ACMN_1000_dist_100(alpha):
    top = Topology.Topology(1)
    top.init_nsf()
    top.create_ACMN_dataset(14, 21, alpha, 1000, distance_realisations=100)


def get_average_link_dist(amount=10):
    for i in range(amount):
        for j in range(100):
            graph = ON.load_graph("ACMN{}_{}".format(i, j))
            avg_link_dist = {"avg_link_dist": [ON.get_avg_distance(graph)]}
            avg_link_dist_df = pd.DataFrame(avg_link_dist)
            avg_link_dist_df.to_json(path_or_buf="Data/_avg_link_dist_{}_{}.json".format(i, j))


def save_data(data, name, location):
    N_lambda_df = pd.DataFrame(data)
    N_lambda_df.to_json(path_or_buf="{}/{}.json".format(location, name))


def route_topologies_MNH():
    sum = 0
    for i in range(1000):
        for j in range(100):
            logging.info("Route Topologies Progress: {}%".format(round(((i + (j / 100)) / 1000) * 100, 3)))
            graph = ON.load_graph("ACMN{}_{}".format(i, j))
            rwa = RWA.RWA(graph)

            # print(ON.RWA_graph[2][3]["weight"])
            rwa.assign_single_lightpaths_MNH()
            N_lambda = rwa.N_lambda
            lambda_max = rwa.wavelength_max
            graph_name = "ACMN{}_{}.weighted.edgelist".format(i, j)
            topology_vector = ON.read_topology_vector_dataset()
            topology_vector = topology_vector["topology_vector"]["{}".format(sum)]
            avg_link_dist = ON.get_avg_distance(graph)
            data = {"N_lambda": [N_lambda], "Total_Throughput": 0, "Avg_Throughput": 0, "lambda_max": [lambda_max],
                    "graph_name": [graph_name],
                    "topology_vector": [topology_vector], "avg_link_distance": [avg_link_dist]}
            save_data(data, "data_{}_{}".format(i, j), location="Data_MNH")
            sum += 1


def route_topologies_SNR():
    sum = 0
    for i in range(1000):
        for j in range(100):
            logging.info("Route Topologies Progress: {}%".format(round(((i + (j / 100)) / 1000) * 100, 3)))
            graph = ON.load_graph("ACMN{}_{}".format(i, j))
            rwa = RWA.RWA(graph)
            rwa.fill_C_band_SNR()
            N_lambda = rwa.N_lambda
            lambda_max = rwa.wavelength_max
            graph_name = "ACMN{}_{}.weighted.edgelist".format(i, j)
            topology_vector = ON.read_topology_vector_dataset()
            topology_vector = topology_vector[i]
            avg_link_dist = ON.get_avg_distance(ON.load_graph("ACMN{}_{}".format(i, j)))
            data = {"N_lambda": [N_lambda], "Total_Throughput": 0, "Avg_Throughput": 0, "lambda_max": [lambda_max],
                    "graph_name": [graph_name],
                    "topology_vector": [topology_vector], "avg_link_distance": [avg_link_dist]}
            logging.info("data: {}".format(data))
            save_data(data, "data_{}_{}".format(i, j), location="Data_SNR")
            sum += 1

def plot_N_lambda_avg_link(amount):
    N_lambda = []
    avg_link_dist = []
    for i in range(amount):
        for j in range(100):
            data = pd.read_json("Data_SNR/data_{}_{}.json".format(amount,j))
            N_lambda.append(data["N_lambda"])
            avg_link_dist.append(data["avg_link_distance"])
    logging.info("N_lambda_sample: {}".format(N_lambda[0:10]))
    logging.info("avg_link_distance: {}".format(avg_link_dist[0:10]))
    plt.plot(avg_link_dist, N_lambda)
    plt.plot()

# once we get to 1% we can plot 10
# once we get to 10% we can plot 100



if __name__ == "__main__":
    #ACMN_1000_dist_100(0.21)
    #route_topologies_SNR()
    plot_N_lambda_avg_link(100)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn.neighbors import KernelDensity
from functools import reduce
import Physical_Layer as pl
import logging
import math
import pandas as pd
from ast import literal_eval

logging.basicConfig(level=logging.INFO)


class Topology():
    def __init__(self, type):
        self.type = type
        self.topology_graph = nx.Graph()

    def init_nsf(self):
        s_N = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 13]
        t_N = [2, 6, 4, 3, 14, 4, 10, 5, 6, 13, 7, 8, 9, 10, 11, 10, 11, 12, 13, 13, 14]
        weights_N = [1520, 1440, 1040, 3520, 2160, 1040, 1040, 1120, 880, 1840, 2960, 880, 1200, 560, 560, 640, 400,
                     1280, 2480, 1520, 2640]
        self.weights = weights_N
        names_N = ['C1', 'WA', 'IL', 'NE', 'CO', 'UT', 'MI', 'NY', 'NJ', 'PA', 'DC', 'GA', 'TX', 'C2']
        num_nodes = 14
        num_edges = len(s_N)
        connectivity = 0
        efficiency = 0
        nodes = np.arange(1, 10)
        self.topology_graph.add_nodes_from(nodes)
        self.topology_graph.add_weighted_edges_from(self.make_weighted_edge_list(s_N, t_N, weights_N))

    def plot_topology(self):
        plt.plot()
        nx.draw(self.topology_graph, with_labels=True)
        plt.show()

    def plot_graph(self, graph):
        plt.plot()
        nx.draw(graph, with_labels=True)
        plt.show()

    def read_topology_vector_dataset(self):
        """This method automatically reads the current saved topology vectors

        :return: topology_vector_dataset
        :rtype: dict"""
        topology_vector_df = pd.read_json(path_or_buf="Data/_topology_vectors.json")

        topology_vector = topology_vector_df["topology_vector"].to_list()

        return topology_vector

    def write_topology_vector_dataset(self):
        """
        This method creates a pandas dataframe from the global topology_vector_dataset and saves as .csv

        :return: None
        :rtype: None
        """

        self.topology_vector_dataset_df = pd.DataFrame(self.topology_vector_dataset)
        self.topology_vector_dataset_df.to_json(path_or_buf="Data/_topology_vectors.json")

    def save_topology(self):
        nx.write_adjlist(self.topology_graph, path="Topology/{}".format(self.type + ".adjlist"))

    def load_topology(self):
        nx.read_adjlist("Topology/{}".format(self.type + ".adjlist"))

    def save_graph(self, graph, name):
        """
        Method that saves graphs as a weighted edge list.

        :param graph: Graph to be saved
        :param name: Name to be saved under in Topolgy Directory
        :return: None
        :rtype: None
        """

        nx.write_weighted_edgelist(graph, path="Topology/{}".format(name + ".weighted.edgelist"))

    def load_graph(self, name):
        """
        Method to load graphs from weighted edge lists stored in Topology

        :param name: Name of graph to be loaded
        :return: returns graph with congestion and NSR assignments
        :rtype: nx.Graph()
        """
        graph = nx.read_weighted_edgelist(path="Topology/{}".format(name + ".weighted.edgelist"),
                                          create_using=nx.Graph(), nodetype=int)
        graph = self.assign_congestion(graph)
        graph = self.assign_NSR(graph)
        return graph

    def make_edge_list(self, s, t):
        edge_list = []
        for i in range(0, len(s)):
            edge_list.append((s[i], t[i]))
        return edge_list

    def make_weighted_edge_list(self, s, t, w):
        edge_list = []
        for i in range(0, len(s)):
            edge_list.append((s[i], t[i], w[i]))

        return edge_list

    def create_ACMN_dataset(self, N, L, alpha, amount, distance_realisations=1):
        """This method is used to create ACMN datasets, which creates the specified amount of topologies randomly, with the specified amount of distance realisations.


        :param N: Number of nodes of ACMN
        :param L: Number of links of ACMN
        :param alpha: Connectivity of graph
        :param amount: Amount of ACMN networks to create
        :param distance_realisations: Amount of network distance realisations to create
        :return: None, saves all topologies to disk in Optical-Networks ->Topology
        :rtype: None"""

        self.topology_vector_dataset = {"topology_vector": []}
        self.write_topology_vector_dataset()
        for i in range(0, amount):
            self.progress = (i / amount) * 100
            ACMN, topology_vector = self.create_ACMN(N, L, alpha, "ACMN{}_0".format(i))
            self.topology_vector_dataset["topology_vector"].append(topology_vector)
            self.write_topology_vector_dataset()
            for j in range(1, distance_realisations):
                ACMN = self.assign_distances(ACMN, scaling_factor=(640 / 1463) + j * 0.0164)
                self.save_graph(ACMN, "ACMN{}_{}".format(i, j))
            logging.info("Progress: {}%".format(int((i / amount) * 100)))
           #print("Progress: {}%".format(int((i / amount) * 100)), end='\r', flush=True)

    def find_degree_vector_ij(self, degree_vector_2D, degree_vector_3D_1, degree_vector_max1, diameter, graph):
        degree_vector_ij = []

        if len(degree_vector_2D) > 1:
            lengths = list(list(nx.shortest_path_length(graph, node[0], degree_vector_2D[i][0]) for i in
                                # create a list of lists with non-repeated lengths between degree node pairs
                                range(0, len(degree_vector_2D))) for node in degree_vector_2D)

            ij1 = list(
                lengths[i][i + 1:] for i in range(0, len(lengths)))  # get the unique values (right from zero diagonl)
            ij1 = list(map(lambda x: x.count(diameter), ij1))  # count instances of paths with hops of length diameter
            ij1 = reduce(lambda x, y: x + y, ij1)  # sum the instances together

            degree_vector_ij.append(
                ij1)  # append them to the ij part of the topology vector (repeat for the others with D-1 and 1 in respect)

        else:
            degree_vector_ij.append(0)
        if len(degree_vector_3D_1) > 1:
            lengths = list(list(nx.shortest_path_length(graph, node[0], degree_vector_3D_1[i][0]) for i in
                                range(0, len(degree_vector_3D_1))) for node in degree_vector_3D_1)

            ij2 = list(lengths[i][i + 1:] for i in range(0, len(lengths)))

            ij2 = list(map(lambda x: x.count(diameter - 1), ij2))
            ij2 = reduce(lambda x, y: x + y, ij2)

            degree_vector_ij.append(ij2)
        else:
            degree_vector_ij.append(0)
        if len(degree_vector_max1) > 1:
            lengths = list(list(nx.shortest_path_length(graph, node[0], degree_vector_max1[i][0]) for i in
                                range(0, len(degree_vector_max1))) for node in degree_vector_max1)

            ij3 = list(lengths[i][i + 1:] for i in range(0, len(lengths)))

            ij1 = list(map(lambda x: x.count(1), ij3))
            ij3 = reduce(lambda x, y: x + y, ij3)

            degree_vector_ij.append(ij3)

        else:
            degree_vector_ij.append(0)
        return degree_vector_ij

    def create_topology_vector(self, graph):
        """
        This method creates the topology vector for a given input graph

        :param graph: the graph from which to derive the topology vector
        :return: topology_vector
        :rtype: List
        """
        d = lambda x: list(x.count(i) for i in range(2, max(x) + 1))  # count number of nodes with same degree

        diameter = nx.diameter(graph)
        degrees = graph.degree()  # Get the degree of all nodes [(node, degree)...]
        degree_vector = list(map(lambda x: x[1], degrees))  # Extract only the degree of the node
        degree_vector_2D = list(filter(lambda x: x[1] is 2, degrees))  # sort degrees into degree of 2
        degree_vector_3D_1 = list(filter(lambda x: x[1] is 2, degrees))  # sort degrees into degree of 3
        degree_vector_max1 = list(filter(lambda x: x[1] is max(degrees), degrees))  # sort degrees into degree of max
        degree_vector_ij = self.find_degree_vector_ij(degree_vector_2D, degree_vector_3D_1, degree_vector_max1,
                                                      diameter,
                                                      graph)  # Go through node pairs of degree two, three and max and see if the are within D, D-1 and max hops

        topology_vector = d(degree_vector) + degree_vector_ij + [diameter] + [
            round(nx.average_shortest_path_length(graph), 2)]  # Concatenate the topology vector

        return topology_vector

    def create_ACMN(self, N, L, alpha, name, plot=False):
        """
        This method creates a randomly generated graph that is subject to constraints C1, C2 and unique topology within the
        _topology_vectors.csv file. It return a graph and topology vector for said graph.

        :param N: Number of Nodes
        :param L: Number of Links
        :param alpha: Connectivity of graph
        :param name: Name to be saved as
        :param plot: plot - False
        :return: graph, topology_vector
        :rtype: nx.Graph(), list
        """
        nodes = np.arange(1, N + 1)  # list with nodes in it
        ACMN = nx.Graph()  # creating graph
        ACMN.add_nodes_from(nodes)  # adding nodes
        alpha_ACMN = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (N_ACMN * (N_ACMN - 1))  # lambda function for connectivity
        link_ACMN = lambda N: (int(np.random.uniform(1, N)), int(
            np.random.uniform(1, N)))  # lambda function for choosing random link from uniform distribution
        while alpha_ACMN(N, ACMN.number_of_edges()) < alpha:
            link = link_ACMN(N + 1)
            if link[0] == link[1]:  # if link is connected to same node pass
                pass
            else:
                ACMN.add_edge(link[0], link[1])  # add new link
        if self.check_C1(ACMN) == True and self.check_C2(ACMN) == True and nx.is_connected(ACMN) == True:  # check constrains C1 and C2
            topology_vector = self.create_topology_vector(ACMN)  # assign topology vector

        else:
            ACMN, topology_vector = self.create_ACMN(N, L, alpha, name)  # if C1 or C2 fails try again

        if not self.check_unique_topology_vector(topology_vector):
            ACMN, topology_vector = self.create_ACMN(N, L, alpha, name)

        ACMN = self.assign_distances(ACMN)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)
        self.save_graph(ACMN, name)

        return ACMN, topology_vector

    def assign_distances(self, graph, scaling_factor=1.0):
        """
        This method is for assigning distances from the kernel density estimation of the NSF net

        :param scaling_factor:
        :param graph: graph to be assigned distances
        :param scaling_factor: factor by which to scale distances by: avg_link_dist/1463
        :return: graph
        :rtype: nx.Graph()
        """
        kde = self.kernel_density_pdf(plot=False)  # get the kernel density estimation of distances from nsf
        samples = kde.sample(len(graph.edges))  # draw samples from estimation
        samples = list(
            map(lambda x: x * scaling_factor, samples))  # scale them by the scaling factor (see API - assign distances)
        if list(map(lambda x: math.ceil(x / 80), samples)) == 0:
            print("zero links")
        distances = list(
            map(lambda x, y: (x[0], x[1], abs(math.ceil((y / 80)))), graph.edges,
                samples))  # assign distances to the edges
        logging.debug("distances: {}".format(distances))
        graph.clear()  # clear graph and assign new vertives
        graph.add_weighted_edges_from(distances)
        # print(graph.edges.data('weight'))
        # self.plot_graph(graph)
        return graph

    def assign_congestion(self, graph):
        congestion = list(map(lambda x: (x[0], x[1], {"congestion": 0}), graph.edges))
        graph.add_edges_from(congestion)
        return graph

    def get_avg_distance(self, graph):
        """
        This method gets the average link distance for a given input graph

        :param graph: Graph to calculate the average link distance for
        :return: average link distance
        :rtype: float
        """
        edges = graph.edges()
        logging.debug("edge attr: {}".format(nx.get_edge_attributes(graph, "weight")))
        edge_dist = list(map(lambda x: graph[x[0]][x[1]]["weight"] * 80, edges))
        logging.debug("edge_dist: {}".format(edge_dist))
        avg_dist = sum(edge_dist) / len(edge_dist)
        return avg_dist

    def assign_NSR(self, graph):
        NSR = list(map(lambda x: (x[0], x[1], {"NSR": 0}), graph.edges))
        graph.add_edges_from(NSR)
        return graph

    def check_C1(self, graph):
        bridges = nx.bridges(graph)
        bridges = list(bridges)
        logging.debug(bridges)
        if len(bridges) != 0:
            return False
        else:
            return True

    def check_C2(self, graph):
        for item in graph.degree:
            logging.debug(item)
            if item[1] < 2:
                # print(False)
                return False

                # print(True)
        return True

    def check_unique_topology_vector(self, topology_vector):
        """

        :param topology_vector: The topology vector to be tested
        :return: True or False - depending on condition
        :rtype: Boolean
        """
        topology_vector_dataset = self.read_topology_vector_dataset()
        logging.debug(topology_vector_dataset)
        if topology_vector in topology_vector_dataset:
            return False
        else:
            return True

    def gaussian_pdf(self, span, bin_size=30, sample_number=1000, plot=True):
        avg = np.mean(self.weights)
        var = np.var(self.weights)
        sd = var ** (0.5)
        pd = np.random.normal(avg, sd, (sample_number))
        if plot == True:
            count, bins, ignored = plt.hist(pd, bin_size, density=True)
            plt.plot(bins, 1 / (sd * np.sqrt(2 * np.pi)) * np.exp((-(bins - avg) ** 2 / (2 * sd ** 2))), linewidth=2,
                     color='r')
            plt.show()
        return pd

    def kernel_density_pdf(self, bandwidth=250, plot=True):
        X = np.asarray(self.weights)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        if plot == True:
            X_plot = np.linspace(0, max(self.weights), 10000)[:, np.newaxis]
            log_dens = kde.score_samples(X_plot)
            plt.hist(self.weights, 70, density=True)
            plt.plot(X_plot, np.exp(log_dens), 'r-', lw=2, alpha=0.6, label='norm pdf')
            plt.show()

        return kde


if __name__ == "__main__":
    nsf_top = Topology("nsf")
    nsf_top.init_nsf()
    # nsf_top.plot_topology()
    # nsf_top.save_topology()
    # nsf_top.load_topology()
    # print(nsf_top.gaussian_pdf(80))
    # nsf_top.kernel_density_pdf()
    # ACMN = nsf_top.create_ACMN(14, 21, 0.2, "test", plot=True)
    # nsf_top.plot_graph(ACMN)
    # nsf_top.create_ACMN_dataset(14, 21, 0.3, 10)
    # logging.info("topology vector database: {}".format(nsf_top.read_topology_vector_dataset()))
    print(nsf_top.load_graph("ACMN0_1"))

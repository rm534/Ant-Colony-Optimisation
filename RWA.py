import networkx as nx
import Topology
import logging
import Physical_Layer
import pandas as pd

logging.basicConfig(level=logging.INFO)


# logger = logging.getLogger()
class LA():
    def __init__(self):
        self.lightpath_routes = []


class WA():
    def __init__(self):
        self.current_wavelength = 0
        self.wavelengths = {0: []}
        self.wavelength_max = 0


class RWA(Physical_Layer.ISRSGN_Model, LA, WA):
    """class that can take in a graph and store RWA methods for this graph.

    :param graph: an input graph generated from Topology
    """

    def __init__(self, graph):
        super().__init__()
        Physical_Layer.ISRSGN_Model.__init__(self)
        WA.__init__(self)
        LA.__init__(self)
        self.RWA_graph = graph
        logging.debug(graph)
        self.SNR_graph = self.add_SNR_links(graph)

    def get_shortest_dijikstra_all(self):  # gets all shortest unique paths in a graph for all node pairs
        shortest_paths = dict(nx.all_pairs_dijkstra_path(self.RWA_graph))
        logging.debug(shortest_paths)
        clone = shortest_paths.copy()
        shortest_unique_paths = []
        for item in shortest_paths:
            for other in shortest_paths[item]:
                logging.debug("before: {}".format(shortest_paths[item][other]))
                # print("before: {}".format(shortest_paths[item][other]))
                sort = sorted(clone[item][other])
                logging.debug("after: {}".format(shortest_paths[item][other]))
                # print("after: {}".format(shortest_paths[item][other]))
                logging.debug(shortest_unique_paths)
                # print(shortest_unique_paths)
                # print(clone[item][other])
                if other == item:
                    pass
                elif sort in shortest_unique_paths:

                    # print(clone[item][other].sort())
                    # print(shortest_unique_paths)
                    pass
                else:
                    shortest_unique_paths.append((item, other, shortest_paths[item][other]))
                    # print(shortest_unique_paths)
                    logging.debug("test: {}".format(shortest_paths[item][other]))
                    # print("test: {}".format(shortest_paths[item][other]))
        logging.debug("shortest_unique_paths: {}".format(shortest_unique_paths))
        shortest_unique_paths = self.remove_duplicates_in_paths(shortest_unique_paths)
        return shortest_unique_paths

    def remove_duplicates_in_paths(self, paths):
        for path in paths:

            for i in range(len(paths) - 1):
                logging.debug(path[2][-1])
                logging.debug(paths[i][2][-1])
                if path[2][0] == paths[i][2][-1] and path[2][-1] == paths[i][2][0] and path[2] != paths[i][2]:
                    logging.debug("path to remove: {}".format(paths[i]))
                    paths.remove(paths[i])
        return paths

    def djikstras_shortest_paths(self):  # CAREFUL: only gets for all edges
        edges = self.RWA_graph.edges()
        # print(edges)
        shortest_paths = list(
            map(lambda edges: (edges[0], edges[1], nx.dijkstra_path(self.RWA_graph, edges[0], edges[1])), edges))
        # shortest_paths2 = list(map(lambda x: ()))
        shortest_paths2 = dict(nx.all_pairs_dijkstra_path(self.RWA_graph))
        logging.debug("shortest paths: {}".format(shortest_paths))
        # print(shortest_paths)
        shortest_paths = self.get_shortest_dijikstra_all()
        return shortest_paths

    def sort_length(self, elem):  # function to sort k_shortest by length of path
        return len(elem[2])

    def sort_cost(self, elem):  # function to sort k_shortest paths by cost input format: ((s, d), cost, [path])
        return elem[1]

    def get_k_shortest_paths_MNH(self, limit=100):
        shortest_paths_all = self.get_shortest_dijikstra_all()  # get all shortest paths for node pairs
        k = 2  # initialising k as 2
        k_shortest_ordered_min = []  # list to hold the equal cost paths for all nodes

        def sort_length(elem):  # function to sort k_shortest by length of path
            return len(elem[2])

        for path in shortest_paths_all:  # for all node pairs
            k = 2
            logging.debug("path: {}".format(path))
            while k <= limit:  # k < set limit in parameters
                k_shortest = self.yens_k_shortest_paths(k, path)  # find the k_shortest paths
                k_shortest = sorted(k_shortest, key=self.sort_length)  # sort these according to length of paths
                k_shortest = list(map(lambda x: x[2], k_shortest))  # take only the paths ([[path] ...]
                logging.debug("k_shortest_ordered: {}".format(k_shortest))
                logging.debug("k_shortest[-1] : k_shortest[0]  {} : {}".format(len(k_shortest[-1]), len(k_shortest[0])))
                if len(k_shortest[-1]) > len(k_shortest[
                                                 0]):  # if first value (shortest path) is smaller in length that path[k] filter the list for paths only as long as shortest path and break
                    y = len(k_shortest[0])
                    k_shortest = list(filter(lambda x: len(x) == y, k_shortest))
                    if self.RWA_graph.has_edge(k_shortest[0][0], k_shortest[0][-1]) and [k_shortest[0][0],
                                                                                         k_shortest[0][
                                                                                             -1]] not in k_shortest:
                        k_shortest.append([k_shortest[0][0], k_shortest[0][-1]])

                    logging.debug("k_shortest_filtered: {}".format(k_shortest))
                    k_shortest_ordered_min.append(((path[0], path[1]), k_shortest))
                    logging.debug("BRAEK!!!")
                    break
                elif len(k_shortest[-1]) == len(k_shortest[0]):
                    if k >= limit:
                        k_shortest_ordered_min.append(((path[0], path[1]),
                                                       k_shortest))  # if path[k] is same length as shortest path increment k by a factor of 2
                    logging.debug("k_shortest_ordered_min: {}".format(k_shortest_ordered_min))
                    k *= 2

            logging.debug("k_shortest_ordered_min: {}".format(k_shortest_ordered_min))
        self.equal_cost_paths = k_shortest_ordered_min

    def get_k_shortest_paths_SNR(self, bandwidth=0.9, limit=50):

        shortest_paths_all = self.get_shortest_dijikstra_all()
        logging.debug("shortest_paths_all: {}".format(shortest_paths_all))
        k_shortest_ordered_min = []
        # k=2
        # get shortest paths

        # choose best SNR path(smallest NSR) and calculate throughput x
        for path in shortest_paths_all:
            k = 2
            while k <= limit:
                k_shortest = self.yens_k_shortest_paths(k,
                                                        path)  # get k_shortest paths with yens algorithm for link lengths
                k_shortest = self.update_path_cost(self.SNR_graph, k_shortest,
                                                   weight="weight")  # update path costs for NSR
                k_shortest = sorted(k_shortest, key=self.sort_cost,
                                    reverse=True)  # sort according to best SNR (smallest NSR is best SNR)
                k_shortest_paths = list(
                    map(lambda x: x[2], k_shortest))  # get the k_paths instead of cost and other info
                logging.debug("k_shortest_paths: {}".format(k_shortest))

                capacity = self.calculate_capacity_lightpath(
                    (1 / k_shortest[-1][1]))  # taking inverse of smallest NSR to calculate capacity with SNR
                C1 = capacity * bandwidth  # criteria C1 - 90% of capacity of best SNR link
                worst_capacity = self.calculate_capacity_lightpath((1 / k_shortest[0][1]))
                logging.debug("C1: {}".format(C1))
                logging.debug("cost of SNR best: {} for path {}".format(1 / k_shortest[-1][1], k_shortest[-1]))
                logging.debug("cost of SNR worst: {} for path {}".format(1 / k_shortest[0][1], k_shortest[0]))
                logging.debug("c[-1]: {}".format(self.calculate_capacity_lightpath((1 / k_shortest[0][1]))))
                if worst_capacity >= C1:  # is largest NSR (worst SNR) above 90% threshold
                    # keep going k *= 2  # if not then it orders values that are above C1 and appends the equal cost paths, otherwise it increments k*=2 and tries again
                    logging.debug(
                        "capacity of lifghtpath: {}".format(self.calculate_capacity_lightpath((1 / k_shortest[0][1]))))
                    if 2 * k > limit:
                        logging.debug("k_shortest_paths: {}".format(k_shortest_paths))
                        logging.debug("answer: {}".format(((path[0], path[1]), k_shortest_paths)))
                        k_shortest_paths = list(
                            filter(lambda x: self.calculate_capacity_lightpath((1 / x[1])) > C1, k_shortest))

                        logging.debug("final k_shortest paths > C1: {}".format(k_shortest_paths))
                        k_shortest_paths = list(map(lambda x: x[2], k_shortest_paths))
                        if [(path[0], path[1])] not in k_shortest_paths and self.RWA_graph.has_edge(path[0], path[1]):
                            k_shortest_paths.append([path[0], path[1]])
                        k_shortest_ordered_min.append(((path[0], path[1]), k_shortest_paths))

                    k *= 2
                else:
                    # filter values of k_shortest paths for only bigger than C2
                    logging.debug("done")
                    logging.debug("answer: {}".format(((path[0], path[1]), k_shortest_paths)))
                    k_shortest_paths = list(
                        filter(lambda x: self.calculate_capacity_lightpath((1 / x[1])) > C1, k_shortest))
                    if len(k_shortest_paths) == 0:
                        # k_shortest_paths.append(k_shortest_paths[2][-1])
                        logging.debug("empty path!!! k_shortest: {}".format(k_shortest_paths))
                    k_shortest_paths = list(map(lambda x: x[2], k_shortest_paths))
                    if [(path[0], path[1])] not in k_shortest_paths and self.RWA_graph.has_edge(path[0], path[1]):
                        k_shortest_paths.append([path[0], path[1]])
                    k_shortest_ordered_min.append(((path[0], path[1]), k_shortest_paths))
                    break
            logging.debug("k_shortest_ordered_min: {}".format(k_shortest_ordered_min))
        self.equal_cost_paths = k_shortest_ordered_min
        return k_shortest_ordered_min

        # is k path throughput < 0.9x?
        # yes end
        # no k*2 and get shortest paths again

    def nodes_to_edges(self, nodes):
        edges = []
        for i in range(0, len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1]))
        return edges

    def yens_k_shortest_paths(self, k, shortest_path):
        # get the shortest paths for the whole graph
        logging.debug("starting shortest paths for each edge: {}".format(shortest_path))
        k_shortest = []  # array for the k shortest paths of the function to return

        source = shortest_path[2][0]  # source of original shortest path
        destination = shortest_path[2][-1]  # destination of original shortest path
        A = [shortest_path[
                 2]]  # initialising k-shortest path list holding the first shortest path[[k-shortest-path] ... []]

        k_shortest.append(
            ((source, destination), self.path_cost(self.RWA_graph, shortest_path[2], weight=True), shortest_path[
                2]))  # append initial shortest path to the final value
        logging.debug("source - destination: {} - {}".format(source, destination))
        for _k in range(1, k):
            logging.debug("k - shortest paths: {} - {}".format(_k, A))

            logging.debug("A: {}".format(A))
            graph_copy = self.RWA_graph.copy()  # copy input graph
            try:
                for i in range(len(A[_k - 1]) - 1):

                    spurNode = A[_k - 1][i]  # spur node is retrieved from the previous k-shortest path (k-1)
                    logging.debug("spur node: {}".format(spurNode))
                    rootpath = A[_k - 1][
                               :i]  # the root path is the sequence of nodes leading from the source to the spur node
                    logging.debug("root path: {}".format(rootpath))
                    removed_edges = []  # empty array to hold removed edges to add back in later
                    for path in A:
                        if len(path) - 1 > i and rootpath == path[
                                                             :i]:  # if the rootpath is the same as in another path in k-shortest paths
                            edge = (path[i], path[i + 1])  # the edge is removed
                            edges = self.nodes_to_edges(rootpath)
                            if not graph_copy.has_edge(edge[0], edge[1]):
                                logging.debug("graph does not have edge : {} {}".format(edge[0], edge[1]))
                                continue
                            removed_edges.append(
                                (path[i], path[i + 1], self.RWA_graph.get_edge_data(path[i], path[i + 1])["weight"]))
                            for item in edges:
                                removed_edges.append(
                                    (item[0], item[1], self.RWA_graph.get_edge_data(item[0], item[1])["weight"]))
                                graph_copy.remove_edge(item[0], item[1])
                            logging.debug("removing edge: ({} {})".format(path[i], path[i + 1]))

                            graph_copy.remove_edge(path[i], path[i + 1])

                    logging.debug("here1")

                    try:
                        logging.debug("here2")
                        spurpath = nx.dijkstra_path(graph_copy, spurNode,
                                                    destination)  # the new shortest path is then found
                        total_path = rootpath + spurpath  # the total path of the new path is addition of rootpath and spurpath
                        logging.debug("here3")
                        logging.debug("total path: {}".format(total_path))
                        logging.debug("removed edges: {}".format(removed_edges))
                        total_path_cost = self.path_cost(self.RWA_graph, total_path,
                                                         weight=True)  # calculating cost of new path
                        k_shortest.append(((shortest_path[0], shortest_path[1]), total_path_cost, total_path))
                        A.append(total_path)
                        for removed_edge in removed_edges:  # adding in the removed edges again
                            logging.debug("removed edge: {}".format(removed_edge))
                            graph_copy.add_weighted_edges_from([removed_edge]
                                                               )

                    except Exception as err:
                        logging.debug("error: {}".format(err))

                        break
            except:
                logging.debug("broken")
                break

            logging.debug("edges: {}".format(self.RWA_graph.edges))
            logging.debug("final k shortest paths: {}".format(k_shortest))

        return k_shortest

    def path_cost(self, graph, path, weight=None):
        pathcost = 0
        for i in range(len(path)):
            if i > 0:
                edge = (path[i - 1], path[i])
                if weight != None:
                    # print(path)
                    # print(graph.edges.data("weight"))
                    # print(graph.get_edge_data(path[i - 1], path[i]))
                    logging.debug(graph.has_edge(path[i - 1], path[i]))
                    pathcost += graph.get_edge_data(path[i - 1], path[i])["weight"]
                else:
                    # just count the number of edges
                    pathcost += 1
        return pathcost

    def update_path_cost(self, graph, paths, weight=None):
        pathcost = 0
        new_paths = []

        for path in paths:
            path_cost = self.path_cost(graph, path[2], weight="weight")
            new_cost = (path[0], path_cost, path[2])
            if new_cost == 0:
                logging.debug("empty cost!!!")
            new_paths.append(new_cost)

        return new_paths

    def SNR_shortest_paths(self):
        shortest_paths = self.djikstras_shortest_paths()

    def add_congestion(self, path):
        path_edges = self.nodes_to_edges(path)
        for edge in path_edges:
            self.RWA_graph[edge[0]][edge[1]]["congestion"] += 1
            # congestion += 1
            # graph.add_edge(edge[0], edge[1], {"congestion": congestion})
            # graph[edge[0]][edge[1]]["congestion"] = congestion
        return self.RWA_graph

    def remove_congestion(self, path):
        path_edges = self.nodes_to_edges(path)
        for edge in path_edges:
            self.RWA_graph[edge[0]][edge[1]]["congestion"] -= 1
            # congestion -= 1
        # graph[edge[0]][edge[1]]["congestion"] = congestion
        return self.RWA_graph

    def static_baroni_MNH_LA(self, iter=1000):
        avg_congestion = []
        equal_cost_paths = self.equal_cost_paths
        current_path = [[[] for i in range(nx.number_of_edges(self.RWA_graph))] for i in
                        range(nx.number_of_edges(self.RWA_graph))]
        logging.debug("hello: {}".format(current_path))
        # current_path = np.zeros((nx.number_of_edges(graph_route), nx.number_of_edges(graph_route))) #create matrix to hold all current lightpaths
        for item in equal_cost_paths:  # loop over all equal cost paths
            logging.debug(item)

            best_SNR_path = item[1][0]  # find best SNR path in equal cost paths
            current_path[item[0][0]][
                item[0][1]] = best_SNR_path  # assign this as initial lightpath for source destination pair
            self.add_congestion(best_SNR_path)  # add congestion for the best SNR path
            subbed = False
        for i in range(0, iter):  # loop over all source destination pairs until no more subs can be made
            subbed = False  # variable to keep track of if anything was able to be substituted
            for item in equal_cost_paths:  # loop over all source destination equal cost paths
                for path in item[1]:  # loop over all possible paths in that source destination pair
                    logging.debug("path: {}".format(path))
                    replace = self.check_congestion(current_path[item[0][0]][item[0][1]],
                                                    path)  # check if path should be replaced
                    if replace:
                        logging.debug("replace!!")
                        logging.debug("old path: {}".format(current_path[item[0][0]][item[0][1]]))
                        self.remove_congestion(current_path[item[0][0]][
                                                   item[0][
                                                       1]])  # remove 1 congestion from all previous links
                        self.add_congestion(path)  # add 1 congestion to all new links
                        current_path[item[0][0]][item[0][1]] = path  # set new current path
                        logging.debug("new path: {}".format(current_path[item[0][0]][item[0][1]]))
                        link = self.get_most_loaded_path_link(path)
                        logging.debug("most congested link: {}".format(link))
                        logging.debug("new congestion: {}".format(self.RWA_graph[link[0]][link[1]]["congestion"]))
                        subbed = True  # smth was substituted
                    else:
                        pass
            if not subbed:
                logging.debug("not subbed")
                break  # break if nothing was subbed
            # logging.info("final routing table: {}".format(graph_route[1][2]))
            avg_congestion.append(self.get_average_congestion())
        df = pd.DataFrame(avg_congestion)
        df.to_csv(path_or_buf="Data\htest.csv")
        logging.debug("graph congestion: {}".format(nx.get_edge_attributes(self.RWA_graph, "congestion")))
        s_d_pairs = list(map(lambda x: x[0], equal_cost_paths))
        logging.debug("lightpath routes: {}".format(current_path[0][3]))
        self.lightpath_routes = current_path
        self.SD_pairs = s_d_pairs
        paths = []
        for edge in self.SD_pairs:
            path = self.lightpath_routes[edge[0]][edge[1]]
            cost = self.path_cost(self.SNR_graph, path, weight=True)
            paths.append((edge, cost, path))
        paths = sorted(paths, key=self.sort_cost, reverse=True)
        self.lightpath_routes_consecutive = paths
        self.lightpath_routes_consecutive_single = paths

    def static_baroni_SNR_LA(self, iter=1000):
        avg_congestion = []
        equal_cost_paths = self.equal_cost_paths
        current_path = [[[] for i in range(nx.number_of_edges(self.RWA_graph))] for i in
                        range(nx.number_of_edges(self.RWA_graph))]
        logging.debug("hello: {}".format(current_path))
        # current_path = np.zeros((nx.number_of_edges(graph_route), nx.number_of_edges(graph_route))) #create matrix to hold all current lightpaths
        for item in equal_cost_paths:  # loop over all equal cost paths
            logging.debug(item)

            best_SNR_path = item[1][-1]  # find best SNR path in equal cost paths
            current_path[item[0][0]][
                item[0][1]] = best_SNR_path  # assign this as initial lightpath for source destination pair
            self.add_congestion(best_SNR_path)  # add congestion for the best SNR path
            subbed = False
        for i in range(0, iter):  # loop over all source destination pairs until no more subs can be made
            subbed = False  # variable to keep track of if anything was able to be substituted
            for item in equal_cost_paths:  # loop over all source destination equal cost paths
                for path in item[1]:  # loop over all possible paths in that source destination pair
                    logging.debug("path: {}".format(path))
                    replace = self.check_congestion(current_path[item[0][0]][item[0][1]],
                                                    path)  # check if path should be replaced
                    if replace:
                        logging.debug("replace!!")
                        logging.debug("old path: {}".format(current_path[item[0][0]][item[0][1]]))
                        self.remove_congestion(current_path[item[0][0]][
                                                   item[0][
                                                       1]])  # remove 1 congestion from all previous links
                        self.add_congestion(path)  # add 1 congestion to all new links
                        current_path[item[0][0]][item[0][1]] = path  # set new current path
                        logging.debug("new path: {}".format(current_path[item[0][0]][item[0][1]]))
                        link = self.get_most_loaded_path_link(path)
                        logging.debug("most congested link: {}".format(link))
                        logging.debug("new congestion: {}".format(self.RWA_graph[link[0]][link[1]]["congestion"]))
                        subbed = True  # smth was substituted
                    else:
                        pass
            if not subbed:
                logging.debug("not subbed")
                break  # break if nothing was subbed
            avg_congestion.append(self.get_average_congestion())
        df = pd.DataFrame(avg_congestion)
        df.to_csv(path_or_buf="Data\htest.csv")
        # logging.info("final routing table: {}".format(graph_route[1][2]))
        logging.debug("graph congestion: {}".format(nx.get_edge_attributes(self.RWA_graph, "congestion")))
        s_d_pairs = list(map(lambda x: x[0], equal_cost_paths))
        logging.debug("lightpath routes: {}".format(current_path[0][3]))
        self.lightpath_routes = current_path
        self.SD_pairs = s_d_pairs
        paths = []
        for edge in self.SD_pairs:
            path = self.lightpath_routes[edge[0]][edge[1]]
            cost = self.path_cost(self.SNR_graph, path, weight=True)
            paths.append((edge, cost, path))
        paths = sorted(paths, key=self.sort_cost, reverse=True)
        self.lightpath_routes_consecutive = paths
        self.lightpath_routes_consecutive_single = paths

    def sort_lightpath_routes_consecutive(self):
        self.lightpath_routes_consecutive = sorted(self.lightpath_routes_consecutive, key=self.sort_cost, reverse=True)

    def assign_wavelengths(self, path):
        path_edges = self.nodes_to_edges(path)
        self.current_wavelength = 0
        assigned = False
        while not assigned:
            assigned = False
            # for lightpath, edge in (wavelengths[current_wavelength], path_edges):
            logging.debug(path)
            for lightpath, edge in ((w1, e1) for w1 in self.wavelengths[self.current_wavelength] for e1 in path_edges):
                # logging.info("lightpath: {} edge: {}".format(lightpath, edge))
                if edge in self.nodes_to_edges(lightpath) or (
                        list(reversed(edge))[0], list(reversed(edge))[1]) in self.nodes_to_edges(lightpath):
                    # logging.info("edge in lightpath")
                    self.current_wavelength += 1
                    assigned = False
                    break
                else:
                    # logging.info("edge not in lightpath")
                    assigned = True
                    pass

            if self.current_wavelength == self.wavelength_max + 1:
                self.wavelength_max += 1
                self.wavelengths[self.wavelength_max] = [path]
                logging.debug("assigned new wavelength")
                break
            elif assigned == True:
                logging.debug("assigned wavelength")
                self.wavelengths[self.current_wavelength].append(path)
        return self.wavelengths, self.wavelength_max

    def convert_path_to_SD_cost_pair_SNR(self, path):
        SD_cost_pair = ((path[0], path[-1]), self.path_cost(self.SNR_graph, path, weight=True), path)
        return SD_cost_pair

    def static_baroni_WA(self):
        self.__init__(self.RWA_graph)
        # graph = self.add_SNR_links(graph.copy())  # creates a SNR version of original graph
        # wavelengths.append((paths[0][0], 0, paths[0][1], paths[0][2]))
        # self.wavelengths = {0: []}
        self.wavelengths[0].append(
            self.lightpath_routes_consecutive[0][2])  # appends first route to the first wavelength
        for i in range(1, len(self.lightpath_routes_consecutive)):  # loop over all lightpaths
            logging.debug("i: {}".format(i))
            path_edges = self.nodes_to_edges(self.lightpath_routes_consecutive[i][2])  # convert path to edges
            logging.debug("path edges: {}".format(path_edges))
            self.current_wavelength = 0
            wavelengths, wavelength_max = self.assign_wavelengths(
                self.lightpath_routes_consecutive[i][2])  # assign a wavelength on first fit principle
            logging.debug(wavelengths)

        logging.debug("wavelengths: {}".format(self.wavelengths))

    def get_most_loaded_path_link(self, path):
        path_edges = self.nodes_to_edges(path)
        cong = list(map(lambda x: self.RWA_graph.get_edge_data(x[0], x[1])["congestion"], path_edges))
        most_cong_link = path_edges[cong.index(max(cong))]
        return most_cong_link

    def check_congestion(self, path_old, path_new):
        replace = False
        most_cong_link_old = self.get_most_loaded_path_link(path_old)
        most_cong_link_new = self.get_most_loaded_path_link(path_new)
        logging.debug("most cong link old: {} most cong link new: {}".format(most_cong_link_old, most_cong_link_new))
        logging.debug(
            "old cong: {} new cong: {} ".format(
                self.RWA_graph[most_cong_link_old[0]][most_cong_link_old[1]]["congestion"],
                self.RWA_graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"]))
        if self.RWA_graph[most_cong_link_old[0]][most_cong_link_old[1]]["congestion"] > \
                self.RWA_graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"] + 1:
            logging.debug(
                "new congestion: {}".format(
                    self.RWA_graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"] + 1))
            replace = True

        else:
            replace = False
        return replace

    def update_congestion(self):
        edges = self.RWA_graph.edges()
        for edge in edges:
            self.RWA_graph[edge[0]][edge[1]]["congestion"] = 0
        for path in self.lightpath_routes_consecutive:
            logging.debug(path)
            self.add_congestion(path[2])
        return self.RWA_graph

    def add_lightpath_all_nodes(self):
        # lightpaths_new = lightpaths.copy()  # copy old lightpaths
        # self.lightpath_routes = lightpaths + lightpaths_new  # add these lightspaths to the old lightpaths to add a symetric set

        lightpaths_new = self.lightpath_routes_consecutive + self.lightpath_routes_consecutive_single
        self.lightpath_routes_consecutive = lightpaths_new
        self.sort_lightpath_routes_consecutive()
        logging.debug("lightpaths_new: {}".format(len(lightpaths_new)))
        self.update_congestion()

        return self.RWA_graph

    def check_C_band(self, graph_copy, c_band_max=156):
        edges = graph_copy.edges()  # get all graph edges
        graph_copy = self.add_SNR_links(graph_copy.copy())
        for edge in edges:  # loop over all edges in the edges of graphs
            logging.debug("edge: {} congestion: {}".format(edge, graph_copy[edge[0]][edge[1]][
                "congestion"]))
            logging.debug("has edge: {}".format(graph_copy.has_edge(edge[0], edge[1])))
            # logging.info("weight of path: {} weight of edge: {}".format(
            #   self.path_cost(self.SNR_graph, self.lightpath_routes[edge[0]][edge[1]], weight=True),
            #  graph_copy[edge[0]][edge[1]]["weight"]))
            # logging.info("path of edge: {}".format(self.lightpath_routes[edge[0]][edge[1]]))
            if graph_copy[edge[0]][edge[1]][
                "congestion"] >= c_band_max:  # if the congestion of this edge(i.e. amount of links traversing this edge)
                return True  # return true (c band is full)
        return False

    # TODO: still uses old graph mechanism and no global mechanism, change this at some point....

    def get_average_congestion(self):
        edges = self.RWA_graph.edges()
        sum_congestion = 0
        for edge in edges:
            sum_congestion += self.RWA_graph[edge[0]][edge[1]]["congestion"]
        avg_congestion = sum_congestion / len(edges)
        return avg_congestion

    def check_zero_congestion(self):
        edges = self.RWA_graph.edges()

        for edge in edges:
            if self.RWA_graph[edge[0]][edge[1]]["congestion"] == 0:
                most_loaded_link = self.get_most_loaded_path_link(self.lightpath_routes[edge[0]][edge[1]])
                logging.debug("zero congestion!!!!")
                logging.debug("edge: {}".format(edge))
                logging.debug("path: {}".format(self.lightpath_routes[edge[0]][edge[1]]))
                logging.debug("most congested of path: {} edge congestion: {}".format(
                    self.RWA_graph[most_loaded_link[0]][most_loaded_link[1]]["congestion"],
                    self.RWA_graph[edge[0]][edge[1]]["congestion"]))
            else:
                logging.debug("congestion non-zero for: {}".format(edge))

    def assign_single_lightpaths_MNH(self):
        """
        This method assigns a single light path for each node pair (not bi-directional).

        :return:
        """
        self.get_k_shortest_paths_MNH()
        self.static_baroni_MNH_LA()
        self.static_baroni_WA()
        self.N_lambda = 1

    def fill_C_band_MNH(self):
        """
        This method is a routine to get the first shortest paths, assign lightpaths for all nodes, initalise the wavelengths
        and then keep adding lightpaths to all nodes until the C band is full.

        :return: None
        :rtype: None
        """
        self.get_k_shortest_paths_MNH()
        self.static_baroni_MNH_LA()
        self.static_baroni_WA()
        self.N_lambda = 1
        while not self.check_C_band(self.RWA_graph):
            self.add_lightpath_all_nodes()
            self.static_baroni_WA()
            self.N_lambda += 1

    def fill_C_band_SNR(self):
        """

        :return:
        """
        self.get_k_shortest_paths_SNR()
        self.static_baroni_SNR_LA()
        self.static_baroni_WA()
        self.N_lambda = 1
        while not self.check_C_band(self.RWA_graph):
            self.add_lightpath_all_nodes()
            self.static_baroni_WA()
            self.N_lambda += 1
    def add_wavelengths_to_links(self):
        """

        :return:
        """
        wavelengths = {}
        graph_edges = self.RWA_graph.edges()
        logging.info(graph_edges)
        logging.info(self.wavelengths)
        for edge in graph_edges:
            wavelengths[edge] = {"wavelengths":[]}
            wavelengths[(edge[1], edge[0])] = {"wavelengths":[]}

        for key in self.wavelengths:
            for path in self.wavelengths[key]:
                edges = self.nodes_to_edges(path)
                for edge in edges:
                    wavelengths[edge]["wavelengths"].append(key)
        nx.set_edge_attributes(self.RWA_graph, wavelengths)
        logging.info(self.RWA_graph[1][8]["wavelengths"])

    def get_lightpath_throughput(self, path):
        self.get

if __name__ == "__main__":
    topology = Topology.Topology("nsf")
    topology.init_nsf()
    # graph = topology.create_ACMN(14, 21, 0.7, "test")
    # rwa = RWA()
    # graph_SNR = rwa.add_SNR_links(graph.copy())
    # k_shortest_paths = rwa.get_node_pairs_all(graph)
    # shortest_path = rwa.get_shortest_dijikstra_all(graph)
    # k_shortest = rwa.yens_k_shortest_paths(graph, 8, shortest_path[0])
    # rwa.get_k_shortest_paths_MNH(graph, limit=50)
    #
    # graph, topology_vector = topology.create_ACMN(14, 21, 0.23, "test")
    graph = topology.load_graph("ACMN0_1")
    rwa = RWA(graph)
    # rwa.get_shortest_dijikstra_all(graph)
    # rwa.get_k_shortest_paths_SNR(bandwidth=0.9)
    rwa.get_k_shortest_paths_SNR()
    # logging.info("graph: {}".format(graph[1][3]["weight"]))
    rwa.static_baroni_SNR_LA(iter=500)
    rwa.static_baroni_WA()
    logging.debug("size of old lightpath routes: {}".format(len(rwa.lightpath_routes_consecutive)))
    rwa.check_C_band(rwa.RWA_graph.copy())
    rwa.check_zero_congestion()
    rwa.get_average_congestion()
    # rwa.add_lightpath_all_nodes()
    # rwa.static_baroni_WA()
    # rwa.add_lightpath_all_nodes()
    # rwa.static_baroni_WA()
    N_lambda = 1
    # logging.info("new lightpaths: {}".format(rwa.lightpath_routes_consecutive))
    # logging.info("wavelengths: {}".format(rwa.wavelengths))
    #while not rwa.check_C_band(graph):
     #   logging.info("size of old lightpath routes: {}".format(len(rwa.lightpath_routes)))
      #  rwa.add_lightpath_all_nodes()
       # logging.info("length of new lightpath routes: {}".format(len(rwa.lightpath_routes)))
        #rwa.static_baroni_WA()
        #N_lambda += 1
    #rwa.check_zero_congestion()
    #logging.info("avg_cong: {}".format(rwa.get_average_congestion()))
    #logging.info("N_lambda: {}".format(N_lambda))
    rwa.add_wavelengths_to_links()

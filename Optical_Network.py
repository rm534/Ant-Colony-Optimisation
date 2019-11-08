import Topology
import RWA
import Physical_Layer
import logging

logging.basicConfig(level=logging.DEBUG)

class OpticalNetwork(Topology.Topology, RWA.RWA):
    def __init__(self, mimic_topology="nsf"):
        Topology.Topology.__init__(self, mimic_topology)
        RWA.RWA.__init__(self, self.topology_graph)
        logging.info("finished setting up Optical Network")


    def init_optical_network(self, mimic_topology="nsf", routing="SNR"):
        self.init_nsf()
        self.graph = self.create_ACMN(14, 21, 0.2, "test")
        #self.get_k_shortest_paths_MNH(self.graph, limit=50)

        if routing == "SNR":

            self.get_k_shortest_paths_SNR()
            self.static_baroni_SNR_LA()
            self.static_baroni_WA()
        logging.info("lighpath routes: {}".format(self.lightpath_routes))
        logging.info("wavelength assignmnets: {}".format(self.wavelengths))



if __name__ == "__main__":
    network = OpticalNetwork()
    network.init_optical_network()



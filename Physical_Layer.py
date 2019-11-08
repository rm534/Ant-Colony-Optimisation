import numpy as np
import networkx as nx
import Topology
import logging


class ISRSGN_Model():
    def __init__(self):
        self.channel_parameters = {"attenuation": 0,
                                   "Cr": 0,
                                   "LaunchPower": 0,
                                   "fi": 0,
                                   "Bandwidth": 0,
                                   "spans": 0,
                                   "length": 0,
                                   "D": 0,
                                   "S": 0,
                                   "gamma": 0,
                                   "RefLambda": 0,
                                   "coherent": 0,
                                   "SPM": 0,
                                   "XPM": 0}
        self.assign_physical_wavelengths()

    def init_parameters(self):
        self.channel_parameters["SPM"] = np.zeros(
            (len(self.channel_parameters["fi"]), len(self.channel_parameters["spans"])))
        self.channel_parameters["XPM"] = np.zeros(
            (len(self.channel_parameters["fi"]), len(self.channel_parameters["spans"])))

    def graph_init(self, graph):
        pass

    def assign_physical_wavelengths(self):
        """

        :return:
        """

        B_cband = 5 * 10 ** 12
        baud_rate = 32 * 10 ** 9
        channels = int(B_cband / baud_rate)
        Cband_width = 35
        channel_spacing = 35 / channels
        wavelengths_physical = []
        for i in range(channels):
            wavelengths_physical.append(1530 + i * channel_spacing)
        self.wavelengths_physical = wavelengths_physical

    def add_SNR(self, channels, N, NF=10 ** (4 / 10), a=0.2, L=80, h=6.62607004 * 10 ** (-34),
                f=(3 * 10 ** 8) / (1550 * 10 ** (-9)), B_ref=32 * 10 ** 9, _lambda=1550 * 10 ** (-9), D=17, gamma=1.3):
        gain = a * L
        g = 10 ** (gain / 10)
        P_ase = NF * (g - 1) * h * f * B_ref
        eta = self.get_eta(B_ref, _lambda, a, L, D, gamma, channels)

        if N == 0:
            N = 1
        P_opt = (P_ase / 2 * eta) ** (1 / 3)
        # print(10*np.log10(P_opt))
        SNR_opt = (P_opt / ((N * P_ase) + (N * eta * P_opt ** 3)))

        # SNR_opt = 20*np.log10(SNR_opt)
        # print(20*np.log10(SNR_opt))

        NS_ratio = 1 / SNR_opt
        if NS_ratio < 1:
            logging.debug("++++++++++++++++++++++++++++++++++++infinite SNR++++++++++++++++++++++++++++++++++")
            logging.debug("NSR: {} P_opt: {} N: {} P_ase: {} eta: {}".format(NS_ratio, P_opt, N, P_ase, eta))
            pass
        return NS_ratio

    def calculate_capacity_lightpath(self, path_cost, Bref=32 * 10 ** 9):
        return 2 * Bref * np.log2(1 + path_cost)

    def calculate_throughput_path(self, path):
        pass

    def add_SNR_links(self, graph):
        graph_links = graph.edges.data("weight")  # [(u, v, weight)....]
        SNR_links = list(map(lambda x: (x[0], x[1], self.add_SNR(channels=156, N=x[2])), graph_links))
        logging.debug(SNR_links)
        graph.add_weighted_edges_from(SNR_links)
        return graph

    def get_eta(self, B_ref, _lambda, a, L, D, gamma, channels):
        c = 3 * 10 ** (8)
        a = a / 2 / 4.3463 / 10 ** (3)
        L = L * 1000
        Leff = (1 - np.exp(-2 * a * L)) / (2 * a)
        beta2 = -D * (10 ** (-12) / 10)
        gamma = gamma / (10 ** 3)
        BWtot = B_ref * channels
        eta = (2 / 3) ** 3 * gamma ** 2 * Leff * Leff * 2 * a * B_ref * np.arcsinh(
            (1 / 2) * np.pi ** 2 * abs(beta2) / 2 / a * BWtot ** 2) / (np.pi * abs(beta2 * (B_ref ** 3)))
        return eta

    def get_SPM_n(self, frequency_i, gamma, bandwidth, C_r, alpha, launch_power, beta_2, beta_3):
        """
        This method calculates the self phase modulation (SPM) noise coefficient for a given channel of interest(COI)

        :param frequency_i: frequency of COI
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param launch_power: power at which channel is launched (dBm)
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :return: SPM coefficient
        ":rtype: float
        """
        phi_i = 12 * np.pi ** 2 * (beta_2 + 2 * np.pi * beta_3 * frequency_i)
        T_i = 2 - ((frequency_i * launch_power * C_r) / alpha)
        A = (gamma ** 2) / (bandwidth ** 2)
        B = (np.pi * (T_i ** 2 - (4 / 9))) / (alpha * phi_i)
        C = np.arcsinh((bandwidth ** 2 * phi_i) / 16 * alpha)
        D = (bandwidth ** 2) / (9 * alpha ** 2)

        n_SPM = (16 / 27) * A * (B * C + D)
        return n_SPM

    def get_XPM_n(self, COI, edge, graph, gamma, bandwidth, C_r, alpha, launch_power, beta_2, beta_3):
        """
        This method calculates the cross phase modulation (XPM) for a given COI and edge in a graph with other
        interfering channels. It does it for variably loaded channels.

        :param COI: Channel of interest (integer - corresponds to the wavelengths used for channel)
        :param edge: Link of graph that is being calculated for
        :param graph: graph that is being considered (nx.Graph())
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param launch_power: power at which channel is launched (dBm)
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :return: XPM coefficient
        :rtype: float
        """
        XPM_contr = 0
        frequency = lambda wavelength: (3 * 10 ** 8) / (wavelength)
        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
        interfering_wavelengths = wavelengths.remove(COI)
        interfering_wavelengths_physical = list(map(lambda x: self.wavelengths_physical[x], interfering_wavelengths))
        COI_wavelength = self.wavelengths_physical[COI]
        interfering_frequency = frequency(interfering_wavelengths_physical)
        COI_frequency = frequency(COI_wavelength)
        A = (gamma ** 2) / alpha
        for k in range(len(interfering_wavelengths)):
            phi_i_k = 2 * np.pi ** 2 * (interfering_frequency[k] - COI_frequency) * (
                    beta_2 + np.pi * beta_3 * (COI_frequency + interfering_frequency[k]))
            B = (1 / (bandwidth * phi_i_k))
            C = (T_k ** 2 - 1) / 3
            D = np.arctan((bandwidth * phi_i_k) / alpha)
            E = ((4 - T_k ** 2) / 6)
            F = np.arctan((bandwidth * phi_i_k) / 2 * alpha)
            XPM_contr += B * (C * D + E * F)
        n_XPM = (32 / 27) * A * XPM_contr
        return n_XPM

    def get_non_linear_coefficient(self, spans, COI, edge, graph, gamma, bandwidth, C_r, alpha, launch_power, beta_2,
                                   beta_3, coherence_factor=0):
        """
        This method calculates the SPM and XPM coefficients and returns the total non-linear coefficient.

        :param spans: amount of spans the link has
        :param COI: Channel of interest (integer - corresponds to the wavelengths used for channel)
        :param edge: Link of graph that is being calculated for
        :param graph: graph that is being considered (nx.Graph())
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param launch_power: power at which channel is launched (dBm)
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :param coherence_factor: (...)
        :return: non-linear coefficient
        :rtype: float
        """
        SPM_n = self.get_SPM_n(frequency_i, gamma, bandwidth, C_r, alpha, launch_power, beta_2, beta_3)
        XPM_n = self.get_XPM_n(COI, edge, graph, gamma, bandwidth, C_r, alpha, launch_power, beta_2, beta_3)
        non_linear_coefficient = SPM_n * spans ** (1 + coherence_factor) + XPM_n * spans
        return non_linear_coefficient

    def get_SNR(self, COI, edge, graph, gamma=1.3, channel_bandwidth=32 * 10 ** 9, C_r=0, alpha=0.2):
        """
        This method calculates the SNR for a given channel on a given link in a given graph.

        :param COI: Channel of interest
        :param edge: link in graph
        :param graph: graph to use
        :param gamma: non-linear coefficient
        :param channel_bandwidth: bandwidth of channel (32 Gbaud)
        :return: SNR value
        :rtype: float
        """
        launch_power = self.get_launch_power()
        spans = graph[edge[0]][edge[1]]["weight"]
        non_linear_coefficient = self.get_non_linear_coefficient(spans, COI, edge, graph, gamma, channel_bandwidth, C_r, alpha, launch_power, )
        SNR = launch_power / (P_ase + non_linear_coefficient * launch_power ** 3)
        return SNR

    def get_P_ase(self):
        pass

    def get_launch_power(self):
        pass


if __name__ == "__main__":
    physical_model = ISRSGN_Model()

    topology = Topology.Topology("nsf")
    topology.init_nsf()

    graph = topology.create_ACMN(14, 21, 0.31, "test")
    # physical_model.add_SNR_links(graph)
    # physical_model.calculate_capacity_lightpath(graph, [1, 2, 3, 4], 2000)
    physical_model.add_SNR_links(graph)

""" Approach for different traffics and calculating the SNR for that
    def SPM(self, phi_i, T_i, B_i, a, a_bar, gamma):
        SPM = 4 / 9 * gamma ** (2) / B_i ** (2) * np.pi / (phi_i * a_bar * (2 * a + a_bar)) * ((
                T_i - a ** (2) / a * np.arcsinh(phi_i * B_i ** (2) / a / np.pi) + (
                (a + a_bar) ** (2) - T_i / (a + a_bar) * np.arcsinh(
            np.divide(phi_i * B_i ** (2), (a + a_bar)) / np.pi))))

    def XPM(self, Pi, Pk, phi_ik, T_k, B_i, B_k, a, a_bar, gamma):
        XPM = 32 / 27 * np.sum((Pk / Pi) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar * (2 * a + a_bar)) * (
                (T_k - a ** 2) / a * np.arctan(phi_ik * B_i / a) + ((a + a_bar) ** 2 - T_k) / (
                a + a_bar) * np.arctan(phi_ik * B_i / (a + a_bar))))

    def ISRSGNmodel(self):

        for j in range(0, len(n)):
            for i in range(0, len(fi)):
                (self.SPM(), self.XPM())
"""

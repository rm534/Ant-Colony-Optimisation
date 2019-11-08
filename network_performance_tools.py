import RWA
import numpy as np
rwa = RWA.RWA()

def calculate_throughput_lightpath(graph, lightpath):
    NSR = rwa.path_cost(lightpath, weight=True)
    SNR = 1/NSR
    capacity = 2*32*10**9*np.log2(1+SNR)
    return capacity

def calculate_throughput_total(graph, lightpaths):
    capacity_sum = 0
    for lightpath in lightpaths:
        capacity_sum+=calculate_throughput_lightpath(graph, lightpath)
    total_throughput = (5*10**12)/(32*10**9)*(1/156)*capacity_sum
    return total_throughput


def calculate_network_diameter(graph):
    pass


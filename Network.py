import itertools
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import random
from scipy import special as math
import Signal_information
import Node
import Line
import Lightpath
import Connection

BER_t = 1e-3

Bn = 12.5e9  # noise bandwidth


class Network(object):

    def __init__(self, json_file, transceiver="fixed_rate"):
        # load the file in a json variable
        nodes_json = json.load(open(json_file, "r"))
        # empty struct dict
        self.nodes = {}
        self.lines = {}
        self._weighted_paths = None
        self._connected = False
        self._route_space = None

        # json file--> dict, load in dict node's label --> init Node
        for node_label in nodes_json:
            node_dict = nodes_json[node_label]
            node_dict["label"] = node_label
            node = Node(node_dict)
            self.nodes[node_label] = node
            if "transceiver" not in nodes_json[node_label].keys():
                node.transceiver = transceiver
            else:
                node.transceiver = nodes_json[node_label]["transceiver"]

            for connect_n_label in node_dict["connected_nodes"]:
                line_dict = {}
                line_label = node_label + connect_n_label
                line_dict["label"] = line_label
                # starting node pos (x, y)
                node_position = np.array(nodes_json[node_label]["position"])

                # pos of the other node(x, y)
                connect_n_pos = np.array(nodes_json[connect_n_label]["position"])

                # length=sqrt((x1-x2)^2+(y1-y2)^2)
                line_dict["length"] = np.sqrt(np.sum(node_position - connect_n_pos) ** 2)

                line = Line(line_dict)
                self.lines[line_label] = line

    @property
    def route_space(self):
        return self._route_space

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def set_weighted_paths(self, s_power):
        if not self._connected:
            self.connect()
        node_labels = self.nodes.keys()
        all_couples = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label2 != label1:
                    all_couples.append(label1 + label2)

        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []
        for couples in all_couples:
            for path in self.find_paths(couples[0], couples[1]):
                path_str = ''
                for node in path:
                    path_str += node + "-->"
                paths.append(path_str[:-3])
                s_i = Signal_information(s_power, path)
                if couples in self.lines.keys():
                    line = self.lines[couples]
                    s_power = line.optimized_launch_power(line.eta_nli(s_i.df, s_i.Rs))
                s_i.set_signal_power(s_power)

                s_i = self.propagate(s_i, occupation=False)
                latencies.append(s_i.getLatency())
                noises.append(s_i.getNoisePower())
                snrs.append(10 * np.log10(s_i.getSignalPower() / s_i.getNoisePower()))

        df["path"] = paths
        df["latency"] = latencies
        df["noise"] = noises
        df["snr"] = snrs
        self._weighted_paths = df

        # set the route space free
        route_space = pd.DataFrame()
        route_space["path"] = paths
        for i in range(10):  # every line has 10 channel
            route_space[str(i)] = ["free"] * len(paths)
        self._route_space = route_space

    def draw(self):
        nodes = self.nodes
        font = {"family": "serif",
                "color": "blue",
                "weight": "normal",
                "size": 15
                }
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0] / 1e3
            y0 = n0.position[1] / 1e3
            plt.plot(x0, y0)

            plt.text(x0, y0, node_label, fontdict=font)

            for connect_n_label in n0.connected_nodes:
                n1 = nodes[connect_n_label]
                x1 = n1.position[0] / 1e3
                y1 = n1.position[1] / 1e3
                plt.plot([x0, x1], [y0, y1])
        plt.xlabel("Km")
        plt.title("Network")
        plt.show()

    def free_space(self):
        states = ["free"] * len(self.route_space["path"])
        for l in self.lines.values():
            l.free_state()
        for i in range(10):
            self.route_space[str(i)] = states

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                       if ((key != label1) & (key != label2))]

        cross_lines = self.lines.keys()
        inner_paths = {"0": label1}
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [
                    inner_path + cross_node
                    for cross_node in cross_nodes
                    if ((inner_path[-1] + cross_node in cross_lines) &
                        (cross_node not in inner_path))]

        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)

        return paths

    def find_best_snr(self, in_node, out_node):
        available_path = self.available_path(in_node, out_node)
        if available_path:
            inout_df = self.weighted_paths.loc[
                self.weighted_paths.path.isin(available_path)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[
                inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, in_node, out_node):
        available_path = self.available_path(in_node, out_node)
        if available_path:
            inout_df = self.weighted_paths.loc[
                self.weighted_paths.path.isin(available_path)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[
                inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None
        return best_path

    # connect the network --> dict
    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        switching_matrix = {}
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                inner_dict = {connected_node: np.zeros(10)}
                for connected_node2 in node.connected_nodes:
                    if connected_node2 != connected_node:
                        dict_tmp = {connected_node2: np.ones(10)}
                        inner_dict.update(dict_tmp)

                switching_matrix.update({connected_node: inner_dict})

                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
            node.switching_matrix = switching_matrix
            switching_matrix = {}

        self._connected = True

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        first_node = self.nodes[path[0]]
        prop_s_i = first_node.propagate(lightpath, occupation)
        return prop_s_i

    #  for each element of connections set the attribute latency or snr (latency by default)
    def stream(self, connections, best="latency"):
        streamed_connections = []
        for connection in connections:
            in_node = connection.in_node
            out_node = connection.out_node
            sig_power = connection.signal_power
            if best == "latency":
                path = self.find_best_latency(in_node, out_node)
            elif best == "snr":
                path = self.find_best_snr(in_node, out_node)
            else:
                print("ERROR INPUT VALUE:", best)
                continue
            if path:
                path_occupancy = self.route_space.loc[
                                     self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy))
                           if path_occupancy[i] == "free"][0]
                lightpath = Lightpath(sig_power, path, channel)
                rb = self.calculate_bit_rate(lightpath, self.nodes[in_node].transceiver)
                if rb == 0:
                    continue
                else:
                    connection.bit_rate = rb
                path_occupancy = self.route_space.loc[
                                     self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy))
                           if path_occupancy[i] == "free"][0]
                path = path.replace("-->", "")
                in_lightpath = Lightpath(sig_power, path, channel)
                out_lightpath = self.propagate(in_lightpath, True)
                connection.latency = out_lightpath.latency
                noise = out_lightpath.noise_power
                connection.snr = 10 * np.log10(in_lightpath.signal_power / noise)
                self.update_route_space(path, channel)
            else:
                connection.latency = 0
                connection.snr = 0
            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace("-->", "")
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    @staticmethod
    def line_set_to_path(line_set):
        path = ""
        elements = list(itertools.permutations(list(line_set), len(list(line_set))))
        for i in range(len(elements)):
            flag = 1
            for j in range(len(elements[i]) - 1):
                if elements[i][j][1] != elements[i][j + 1][0]:
                    flag = 0
                j += 2
            if flag == 1:
                for j in range(len(elements[i])):
                    path += elements[i][j][0]
                return path

    def update_route_space(self, path, channel):
        all_paths = [self.path_to_line_set(p)
                     for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = "occupied"

                path_to_update = self.line_set_to_path(line_set)

                for j in range(len(path_to_update)):
                    if j not in (0, len(path_to_update) - 1):
                        if ((path_to_update[j - 1] in self.nodes[path_to_update[j]].connected_nodes) & (
                                path_to_update[j + 1] in self.nodes[path_to_update[j]].connected_nodes)):
                            self.nodes[path_to_update[j]].switching_matrix[path_to_update[j - 1]][
                                path_to_update[j + 1]][
                                channel] = 0

        self.route_space[str(channel)] = states

    def available_path(self, in_node, out_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1e-3)
        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == in_node) and (path[-1] == out_node))]
        available_path = []
        for path in all_paths:
            path_occupancy = self.route_space.loc[
                                 self.route_space.path == path].T.values[1:]
            if "free" in path_occupancy:
                available_path.append(path)
            return available_path

    def calculate_bit_rate(self, lightpath, strategy):
        global BER_t
        Rs = lightpath.Rs
        global Bn
        path = lightpath.path
        Rb = 0
        GSNR_db = pd.array(self.weighted_paths.loc[self.weighted_paths['path'] == path]['snr'])[0]
        GSNR = 10 ** (GSNR_db / 10)
        if strategy == "fixed_rate":
            if GSNR > 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                Rb = 100
            else:
                Rb = 0

        if strategy == "flex_rate":
            if GSNR < 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                Rb = 0
            elif (GSNR > 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn)) & (GSNR < (14 / 3) * math.erfcinv(
                    (3 / 2) * BER_t) ** 2 * (Rs / Bn)):
                Rb = 100
            elif (GSNR > (14 / 3) * math.erfcinv((3 / 2) * BER_t) ** 2 * (Rs / Bn)) & (GSNR < 10 * math.erfcinv(
                    (8 / 3) * BER_t) ** 2 * (Rs / Bn)):
                Rb = 200
            elif GSNR > 10 * math.erfcinv((8 / 3) * BER_t) ** 2 * (Rs / Bn):
                Rb = 400

        if strategy == "shannon":
            Rb = 2 * Rs * np.log2(1 + Bn / Rs * GSNR) / 1e9

        return Rb

    def node_to_number(self, str):
        nodes = list(self.nodes.keys())
        nodes.sort()
        return nodes.index(str)

    def upgrade_traffic_matrix(self, mtx, nodeA, nodeB):
        A = self.node_to_number(nodeA)
        B = self.node_to_number(nodeB)
        connection = Connection(nodeA, nodeB, 1e-3)
        list_con = [connection]
        self.stream(list_con)
        btr = connection.bit_rate
        if btr == 0:
            mtx[A][B] = float("inf")
            return float("inf")
        mtx[A][B] -= btr
        return mtx[A][B]

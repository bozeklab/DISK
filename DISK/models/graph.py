import numpy as np
import importlib.util
import logging

#######################################################################################################################
### Adapted from: https://github.com/yysijie/st-gcn
#######################################################################################################################


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        Adapted from: https://github.com/yysijie/st-gcn
        For more information, please refer to the section 'Partition Strategies'
            in the paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
                         mouse, fly, rat, mocap

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 file,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(file)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, file):
        try:
            spec = importlib.util.spec_from_file_location("module.name", file)
            skeleton_inputs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(skeleton_inputs)
        except Exception as e:
            print(e)
            raise Exception(f'[ERROR IN GRAPH] Problem with skeleton file {file}')

        self.num_node = skeleton_inputs.num_keypoints
        self.center = skeleton_inputs.center
        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbor_link = []
        self.neighbor_link_color = []
        for i in range(len(skeleton_inputs.neighbor_links)):
            if type(skeleton_inputs.neighbor_links[i][0]) == tuple:
                self.neighbor_link_color.extend([skeleton_inputs.link_colors[i]] * len(skeleton_inputs.neighbor_links[i]))
                self.neighbor_link.extend(skeleton_inputs.neighbor_links[i])
            else:
                self.neighbor_link_color.append(skeleton_inputs.link_colors[i])
                self.neighbor_link.append(skeleton_inputs.neighbor_links[i])

        assert len(self.neighbor_link) == len(self.neighbor_link_color)
        logging.info(f'Loaded skeleton with links {self.neighbor_link} and colors {self.neighbor_link_color}')

        self.edge = self_link + self.neighbor_link

        # MOUSE
        # """
        # [0 'left_hip',
        #  1 'right_hip',
        #  2 'left_coord',  # closer to the center of mass
        #  3 'right_coord',
        #  4 'left_back',  # shoulderblades, seem more mobile
        #  5 'right_back',
        #  6 'left_knee',
        #  7 'right_knee']
        #  """
        # self.neighbor_link = [(0, 1),  # hip link
        #                       (0, 2), (0, 6), (2, 4),  # left side
        #                       (1, 3), (1, 7), (3, 5),  # right side
        #                       (4, 5)  # shoulder link
        #                       ]
        # self.neighbor_link_color = ['blueviolet',  # hip link
        #                              'seagreen', 'seagreen', 'seagreen',  # left side
        #                              'cornflowerblue', 'cornflowerblue', 'cornflowerblue',  # right side
        #                              'blueviolet'  # shoulder link
        #                             ]

        # RAT
        # keypoints = [0 'HeadF', 1 'HeadB', 2 'HeadL',
        #              3 'SpineF', 4 'SpineM', 5 'SpineL',
        #              6 'Offset1', 7 'Offset2',
        #              8 'HipL', 9 'HipR',
        #              10 'ElbowL', 11 'ArmL', 12 'ShoulderL',
        #              13 'ShoulderR', 14 'ElbowR', 15 'ArmR',
        #              16 'KneeR', 17 'KneeL', 18 'ShinL', 19 'ShinR']
        # self.neighbor_link = [(0, 1), (0, 2), (1, 2),  # head links
        #                       (1, 3), (3, 4), (4, 5),  # head to spine + spin links
        #                       (3, 6), (4, 6), (4, 7), (6, 7), (5, 7),  # links to offset
        #                       (5, 8), (5, 9),  # spin to hips
        #                       (3, 12), (12, 10), (11, 10),  # left arm
        #                       (3, 13), (13, 14), (14, 15),  # right arm
        #                       (9, 16), (16, 19),  # right leg
        #                       (8, 17), (17, 18)  # left leg
        #                       ]
        # self.neighbor_link_color = ['orange', 'orange', 'orange',  # head links
        #                             'gold', 'gold', 'gold',  # head to spine + spin links
        #                             'grey', 'grey', 'grey', 'grey', 'grey',  # links to offset
        #                             'gold', 'gold',   # spin to hips
        #                             'cornflowerblue', 'cornflowerblue', 'cornflowerblue',  # left arm
        #                             'turquoise', 'turquoise', 'turquoise',  # right arm
        #                              'hotpink', 'hotpink',   # right leg
        #                              'purple', 'purple',  # left leg
        #                             ]
        # self.center = 4

        # FLY 38 keypoints
        # cf https://github.com/NeLy-EPFL/DeepFly3D/blob/master/images/named_keypoints_left.png
        # cf https://github.com/NeLy-EPFL/DeepFly3D/blob/master/images/named_keypoints_right.png
        # self.neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4),  # front right leg
        #                       (5, 6), (6, 7), (7, 8), (8, 9),  # second right leg
        #                       (10, 11), (11, 12), (12, 13), (13, 14),  # rear right leg
        #                       (34, 16), (16, 17), (17, 18),  # head - abdomen, view on the right
        #                       (15, 34), (16, 35), (17, 36), (18, 37),
        #                       # head abdomen - links between right and left views
        #                       (15, 35), (35, 36), (36, 37),  # head - abdomen, view on the left
        #                       (19, 20), (20, 21), (21, 22), (22, 23),  # front left leg
        #                       (24, 25), (25, 26), (26, 27), (27, 28),  # second left leg
        #                       (29, 30), (30, 31), (31, 32), (32, 33),  # rear left leg
        #                       (15, 0), (0, 5), (5, 10), (10, 35),
        #                       # connect head to legs to abdomen on the right side
        #                       (34, 19), (19, 24), (24, 29), (29, 16)
        #                       # connect head to legs to abdomen on the left side
        #                       ]
        # self.neighbor_link_color = ['orange', 'orange', 'orange', 'orange',
        #                             'gold', 'gold', 'gold', 'gold',
        #                             'grey', 'grey', 'grey', 'grey', 'grey',
        #                             'gold', 'gold', 'gold',
        #                             'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue',
        #                             'turquoise', 'turquoise', 'turquoise',
        #                             'hotpink', 'hotpink', 'hotpink', 'hotpink',
        #                             'purple', 'purple',  'purple', 'purple',
        #                             'blue', 'blue', 'blue', 'blue',
        #                             'seagreen', 'seagreen', 'seagreen', 'seagreen',
        #                             'darkolivegreen', 'darkolivegreen', 'darkolivegreen', 'darkolivegreen'
        #                             ]
        #
        # self.center = 16

        # Mocap Human
        # cf https://www.researchgate.net/publication/329899970_Riemannian_Spatio-Temporal_Features_of_Locomotion_for_Individual_Recognition/figures?lo=1
        # keypoints = [0 'spine_0', 1 'spine_1', 2 'spine_2', 3 'spine_3' (top),
        #              4 'arm1_0', 5 'arm1_1', 6 'arm1_2', 7 'arm1_3',
        #              8 'arm2_0', 9 'arm2_1', 10 'arm2_2', 11 'arm2_3',
        #              12 'leg1_0', 13 'leg1_1', 14 'leg1_2', 15 'leg1_3',
        #              16 'leg2_0', 17 'leg2_1', 18 'leg2_2', 19 'leg2_3']
        # self.neighbor_link = [(0, 1), (1, 2), (2, 3),  # spline from bottom to top
        #                       (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
        #                       (0, 12), (12, 13), (13, 14), (14, 15),  # left leg
        #                       (3, 8), (8, 9), (9, 10), (10, 11),  # right arm
        #                       (3, 4), (4, 5), (5, 6), (6, 7)  # left arm
        #                       ]
        # self.center = 0  # lower spine
        # self.neighbor_link_color = ['gold', 'gold', 'gold',  # spline from bottom to top
        #                             'cornflowerblue', 'cornflowerblue', 'cornflowerblue',  'cornflowerblue',  # right leg
        #                             'turquoise', 'turquoise', 'turquoise', 'turquoise',  # left leg
        #                              'hotpink', 'hotpink', 'hotpink', 'hotpink',  # right arm
        #                              'purple', 'purple', 'purple', 'purple',  # left arm
        #                             ]

        # FISH
        # keypoints = ['fish1_head', 'fish1_pec_', 'fish1_tail',
        #              'fish2_head', 'fish2_pec', 'fish2_tail']
        # self.neighbor_link = [(0, 1), (1, 2),  # fish 1
        #                       (3, 4), (4, 5)  # fish 2
        #                       ]
        # self.neighbor_link_color = ['blueviolet', 'cornflowerblue', # fish1
        #                              'seagreen', 'darkolivegreen' # fish2
        #                             ]


    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

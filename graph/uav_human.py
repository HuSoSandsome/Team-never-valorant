# import sys
# import numpy as np

# sys.path.extend(['../'])
# from graph import tools

# num_node = 17 # hrnet
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
#                 (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
#                 (1, 0), (3, 1), (2, 0), (4, 2)]
# inward = [(i, j) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward

# class Graph:
#     def __init__(self, labeling_mode='spatial'):
#         self.num_node = num_node
#         self.self_link = self_link
#         self.inward = inward
#         self.outward = outward
#         self.neighbor = neighbor
#         self.A = self.get_adjacency_matrix(labeling_mode)

#     def get_adjacency_matrix(self, labeling_mode=None):
#         if labeling_mode is None:
#             return self.A
#         if labeling_mode == 'spatial':
#             A = tools.get_spatial_graph(num_node, self_link, inward, outward)
#         else:
#             raise ValueError()
#         return A

  
import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

num_node = 17  
self_link = [(i, i) for i in range(num_node)]  # 自身连接

# 定义骨架关节点的连接
inward = [
   (10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)
]

# 反向连接
outward = [(j, i) for (i, j) in inward]  # 反向连接
neighbor = inward + outward  # 所有相邻节点的连接

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError("Unknown labeling mode")
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print("邻接矩阵:")
    print(A)


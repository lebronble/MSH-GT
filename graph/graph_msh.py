from audioop import reverse
import sys
import numpy as np
sys.path.extend(['../'])
from graph import tools_msh

num_node = 16

class Graph:
    def __init__(self, CoM=0, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools_msh.get_hierarchical_graph(num_node, tools_msh.get_edgeset( CoM=self.CoM)) # L, 3, 16, 16
        else:
            raise ValueError()
        return A,self.CoM


if __name__ == '__main__':
    import tools
    g = Graph().A
    import matplotlib.pyplot as plt
    for i, g_ in enumerate(g[0]):
        plt.imshow(g_, cmap='gray')
        cb = plt.colorbar()
        plt.savefig('./graph_{}.png'.format(i))
        cb.remove()
        plt.show()
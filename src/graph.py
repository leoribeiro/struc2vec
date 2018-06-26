#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

from io import open
from time import time
from six import iterkeys
from collections import defaultdict, Iterable


class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        # logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

        # self.remove_self_loops()

        return self

    def degree(self, nodes=None):
        """
        When given a node, returns the degree.
        When given an iterable of nodes, returns a dictionary mapping nodes to degrees.
        """
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def number_of_edges(self):
        """
        Returns the number of edges in the graph
        """
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        """
        Returns the number of nodes in the graph
        """
        return len(self)

    def to_dict(self):
        d = {}
        for k, v in self.iteritems():
            d[k] = v
        return d


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            if (len(l.strip().split()[:2]) > 1):
                x, y = l.strip().split()[:2]
                x = int(x)
                y = int(y)
                G[x].append(y)
                if undirected:
                    G[y].append(x)
            else:
                x = l.strip().split()[:2]
                x = int(x[0])
                G[x] = []

    G.make_consistent()
    return G

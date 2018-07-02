#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

from io import open
from six import iterkeys
from algorithms import *
from algorithms_distances import *


class Graph():
    def __init__(self, d, is_directed, workers, until_layer=None):

        self.G = d
        self.num_vertices = number_of_nodes_(d)
        self.num_edges = number_of_edges_(d, is_directed)
        self.is_directed = is_directed
        self.workers = workers
        self.calc_until_layer = until_layer
        logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
        logging.info('Graph - Number of edges: {}'.format(self.num_edges))

    def preprocess_neighbors_with_bfs(self):

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(exec_bfs, self.G, self.workers, self.calc_until_layer)

            job.result()

        return

    def preprocess_neighbors_with_bfs_compact(self):

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(exec_bfs_compact, self.G, self.workers, self.calc_until_layer)

            job.result()

        return

    def create_vectors(self):
        logging.info("Creating degree vectors...")
        degrees = {}
        degrees_sorted = set()
        G = self.G
        for v in G.keys():
            degree = len(G[v])
            degrees_sorted.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree]['vertices'] = deque()
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        l = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if index > 0:
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if index < (l - 1):
                degrees[degree]['after'] = degrees_sorted[index + 1]
        logging.info("Degree vectors created.")
        logging.info("Saving degree vectors...")
        save_variable_on_disk(degrees, 'degrees_vector')

    def calc_distances_all_vertices(self, compact_degree=False):

        logging.info("Using compactDegree: {}".format(compact_degree))
        if self.calc_until_layer:
            logging.info("Calculations until layer: {}".format(self.calc_until_layer))

        futures = {}

        vertices = list(reversed(sorted(self.G.keys())))

        if compact_degree:
            logging.info("Recovering degreeList from disk...")
            degree_list = restore_variable_from_disk('compactDegreeList')
        else:
            logging.info("Recovering compactDegreeList from disk...")
            degree_list = restore_variable_from_disk('degreeList')

        parts = self.workers
        chunks = partition(vertices, parts)

        t0 = time()

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            part = 1
            for c in chunks:
                logging.info("Executing part {}...".format(part))
                list_v = []
                for v in c:
                    list_v.append([vd for vd in degree_list.keys() if vd > v])
                job = executor.submit(calc_distances_all, c, list_v, degree_list, part, compact_degree=compact_degree)
                futures[job] = part
                part += 1

            logging.info("Receiving results...")

            for job in as_completed(futures):
                job.result()
                r = futures[job]
                logging.info("Part {} Completed.".format(r))

        logging.info('Distances calculated.')
        t1 = time()
        logging.info('Time : {}m'.format((t1 - t0) / 60))

        return

    def calc_distances(self, compact_degree=False):

        logging.info("Using compactDegree: {}".format(compact_degree))
        if self.calc_until_layer:
            logging.info("Calculations until layer: {}".format(self.calc_until_layer))

        futures = {}

        G = self.G
        vertices = G.keys()

        parts = self.workers
        chunks = partition(vertices, parts)

        with ProcessPoolExecutor(max_workers=1) as executor:

            logging.info("Split degree List...")
            part = 1
            for c in chunks:
                job = executor.submit(split_degree_list, part, c, G, compact_degree)
                job.result()
                logging.info("degreeList {} completed.".format(part))
                part += 1

        with ProcessPoolExecutor(max_workers=self.workers) as executor:

            part = 1
            for c in chunks:
                logging.info("Executing part {}...".format(part))
                job = executor.submit(calc_distances, part, compact_degree=compact_degree)
                futures[job] = part
                part += 1

            logging.info("Receiving results...")
            for job in as_completed(futures):
                job.result()
                r = futures[job]
                logging.info("Part {} completed.".format(r))

        return

    def create_distances_network(self):

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_distances_network, self.workers)

            job.result()

        return

    def preprocess_parameters_random_walk(self):

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_parameters_random_walk)

            job.result()

        return

    def simulate_walks(self, num_walks, walk_length):

        # for large graphs, it is serially executed, because of memory use.
        if len(self.G) > 500000:

            with ProcessPoolExecutor(max_workers=1) as executor:
                job = executor.submit(generate_random_walks_large_graphs, num_walks, walk_length, self.workers,
                                      self.G.keys())

                job.result()

        else:

            with ProcessPoolExecutor(max_workers=1) as executor:
                job = executor.submit(generate_random_walks, num_walks, walk_length, self.workers, self.G.keys())

                job.result()

        return


def load_edgelist(file_, undirected=True):
    """
    Loads an edgelist into a dictionary of (outgoing) neighbours.
    Undirected graphs represented as *symmetric* directed graphs.
    """
    d = {}
    with open(file_) as f:
        for l in f:
            if len(l.strip().split()[:2]) > 1:
                x, y = l.strip().split()[:2]
                x = int(x)
                y = int(y)

                if x not in d:
                    d[x] = []
                if y not in d:
                    d[y] = []

                d[x].append(y)

                if undirected:
                    d[y].append(x)

            else:
                x = l.strip().split()[:2]
                x = int(x[0])
                if x not in d:
                    d[x] = []

    return remove_duplicates_(d)


def remove_duplicates_(graph_dict):
    """
    Remove duplicates in the neighbourhood lists.
    """
    d = {}
    for k in iterkeys(graph_dict):
        d[k] = sorted(set(graph_dict[k]))  # sorted: returns a list
    return d


def number_of_nodes_(graph_dict):
    """
    Returns the number of nodes in a graph represented by a dictionary.
    """
    return len(graph_dict)


def number_of_edges_(graph_dict, is_directed):
    """
    Returns the number of edges in a graph represented by a dictionary.
    """
    degree_sum = sum(len(graph_dict[node]) for node in graph_dict.keys())
    return degree_sum if is_directed else degree_sum / 2

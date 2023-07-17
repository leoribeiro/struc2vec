#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse, logging
import graph
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(filename='struc2vec.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')


def parse_args():
    """
    Parses the struc2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run struc2vec.")

    parser.add_argument('--input', metavar='EDGELIST_FILE', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', metavar='EMB_FILE', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--embed-subset', metavar='NODELIST_FILE', type=str, default=None,
                        help='Only compute embeddings for nodes in a given nodelist. '
                             'Still uses the entire graph as context for the embedding.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--bfs-workers', type=int, default=None,
                        help='Number of parallel workers only for BFS stage. Default is using --workers.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool,
                        help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool,
                        help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                        help='optimization 3')
    return parser.parse_args()


def read_graph(args):
    """
    Reads the input network.
    """
    logging.info(" - Loading graph...")
    graph_dict, in_degrees, out_degrees = graph.load_edgelist(args.input, args.directed, args.weighted)
    logging.info(" - Graph loaded.")
    return graph_dict, in_degrees, out_degrees


def read_embedding_set(args):
    """
    Reads a nodelist with vertices to be embedded.
    """
    vertices = set()
    with open(args.embed_subset, 'r') as f:
        for line in f:
            vertex = int(line.strip())
            vertices.add(vertex)
    return list(vertices)


def learn_embeddings():
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                     workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    logging.info("Representations created.")

    return


def exec_struc2vec(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    if args.weighted and not args.directed:
        raise NotImplementedError('edge weights are only implemented for directed graphs')

    if args.OPT3:
        until_layer = args.until_layer
    else:
        until_layer = None

    if args.embed_subset:
        embedding_vertices = read_embedding_set(args)
    else:
        embedding_vertices = None

    graph_dict, in_degrees, out_degrees = read_graph(args)  # in_degrees = out_degrees = {} if not args.directed
    G = graph.Graph(graph_dict, args.directed, args.workers, bfs_workers=args.bfs_workers, until_layer=until_layer,
                    in_degrees=in_degrees,
                    out_degrees=out_degrees, embedding_vertices=embedding_vertices)

    if args.OPT1:
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if args.OPT2:
        G.create_vectors()
        G.calc_distances(compact_degree=args.OPT1)
    else:
        G.calc_distances_all_vertices(compact_degree=args.OPT1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()

    G.simulate_walks(args.num_walks, args.walk_length)

    return G


def main(args):
    exec_struc2vec(args)

    learn_embeddings()


if __name__ == "__main__":
    args = parse_args()
    main(args)

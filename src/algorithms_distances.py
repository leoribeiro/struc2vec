# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
import math
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
import os


def get_degree_lists_vertices(g, vertices, calcUntilLayer):
    degree_list = {}

    for v in vertices:
        degree_list[v] = get_degree_lists(g, v, calcUntilLayer)

    return degree_list


def get_compact_degree_lists_vertices(g, vertices, calc_until_layer):
    degree_list = {}

    for v in vertices:
        degree_list[v] = get_compact_degree_lists(g, v, calc_until_layer)

    return degree_list


def get_compact_degree_lists(g, root, calc_until_layer):
    t0 = time()

    lists = {}
    vectors = [0] * (max(g) + 1)

    queue = deque()
    queue.append(root)
    vectors[root] = 1
    l = {}

    depth = 0
    pending_depth_increase = 0
    time_to_depth_increase = 1

    while queue:
        vertex = queue.popleft()
        time_to_depth_increase -= 1

        d = len(g[vertex])
        if d not in l:
            l[d] = 0
        l[d] += 1

        for v in g[vertex]:
            if vectors[v] == 0:
                vectors[v] = 1
                queue.append(v)
                pending_depth_increase += 1

        if time_to_depth_increase == 0:

            list_d = []
            for degree, freq in l.iteritems():
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            lists[depth] = np.array(list_d, dtype=np.int32)

            l = {}

            if calc_until_layer == depth:
                break

            depth += 1
            time_to_depth_increase = pending_depth_increase
            pending_depth_increase = 0

    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root, (t1 - t0)))

    return lists


def get_degree_lists(g, root, calc_until_layer):
    t0 = time()

    lists = {}
    vectors = [0] * (max(g) + 1)

    queue = deque()
    queue.append(root)
    vectors[root] = 1

    l = deque()

    depth = 0
    pending_depth_increase = 0
    time_to_depth_increase = 1

    while queue:
        vertex = queue.popleft()
        time_to_depth_increase -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if vectors[v] == 0:
                vectors[v] = 1
                queue.append(v)
                pending_depth_increase += 1

        if time_to_depth_increase == 0:

            lp = np.array(l, dtype='float')
            lp = np.sort(lp)
            lists[depth] = lp
            l = deque()

            if calc_until_layer == depth:
                break

            depth += 1
            time_to_depth_increase = pending_depth_increase
            pending_depth_increase = 0

    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root, (t1 - t0)))

    return lists


def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return ((m / mi) - 1)


def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def preprocess_degree_lists():
    logging.info("Recovering degreeList from disk...")
    degree_list = restore_variable_from_disk('degreeList')

    logging.info("Creating compactDegreeList...")

    dList = {}
    d_frequency = {}
    for v, layers in degree_list.iteritems():
        d_frequency[v] = {}
        for layer, degreeListLayer in layers.iteritems():
            d_frequency[v][layer] = {}
            for degree in degreeListLayer:
                if degree not in d_frequency[v][layer]:
                    d_frequency[v][layer][degree] = 0
                d_frequency[v][layer][degree] += 1
    for v, layers in d_frequency.iteritems():
        dList[v] = {}
        for layer, frequencyList in layers.iteritems():
            list_d = []
            for degree, freq in frequencyList.iteritems():
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d, dtype='float')

    logging.info("compactDegreeList created!")

    save_variable_on_disk(dList, 'compactDegreeList')


def verify_degrees(degree_v_root, degree_a, degree_b):
    if degree_b == -1:
        degree_now = degree_a
    elif degree_a == -1:
        degree_now = degree_b
    elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now


def get_vertices(v, degree_v, degrees, a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices, 2)
    vertices = deque()

    try:
        c_v = 0

        for v2 in degrees[degree_v]['vertices']:
            if v != v2:
                vertices.append(v2)
                c_v += 1
                if c_v > a_vertices_selected:
                    raise StopIteration

        if 'before' not in degrees[degree_v]:
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if 'after' not in degrees[degree_v]:
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if degree_b == -1 and degree_a == -1:
            raise StopIteration
        degree_now = verify_degrees(degree_v, degree_a, degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > a_vertices_selected:
                        raise StopIteration

            if degree_now == degree_b:
                if 'before' not in degrees[degree_b]:
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if 'after' not in degrees[degree_a]:
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']

            if degree_b == -1 and degree_a == -1:
                raise StopIteration

            degree_now = verify_degrees(degree_v, degree_a, degree_b)

    except StopIteration:
        return list(vertices)

    return list(vertices)


def split_degree_list(part, c, G, compact_degree):
    if compact_degree:
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restore_variable_from_disk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restore_variable_from_disk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restore_variable_from_disk('degrees_vector')

    degree_lists_selected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v, len(G[v]), degrees, a_vertices)
        vertices[v] = nbs
        degree_lists_selected[v] = degreeList[v]
        for n in nbs:
            degree_lists_selected[n] = degreeList[n]

    save_variable_on_disk(vertices, 'split-vertices-' + str(part))
    save_variable_on_disk(degree_lists_selected, 'split-degreeList-' + str(part))


def calc_distances(part, compact_degree=False):
    vertices = restore_variable_from_disk('split-vertices-' + str(part))
    degreeList = restore_variable_from_disk('split-degreeList-' + str(part))

    distances = {}

    if compact_degree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1, nbs in vertices.iteritems():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}

            for layer in range(0, max_layer):
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)

                distances[v1, v2][layer] = dist

            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1, v2, (t11 - t00)))

    preprocess_consolides_distances(distances)
    save_variable_on_disk(distances, 'distances-' + str(part))
    return


def calc_distances_all(vertices, list_vertices, degreeList, part, compact_degree=False):
    distances = {}
    cont = 0

    if compact_degree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}

            for layer in range(0, max_layer):
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                distances[v1, v2][layer] = dist

        cont += 1

    preprocess_consolides_distances(distances)
    save_variable_on_disk(distances, 'distances-' + str(part))
    return


def preprocess_consolides_distances(distances, startLayer=1):
    logging.info('Consolidating distances...')

    for vertices, layers in distances.iteritems():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers), startLayer)
        for layer in range(0, startLayer):
            keys_layers.pop(0)

        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')


def exec_bfs_compact(G, workers, calc_until_layer):
    futures = {}
    degree_list = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices, parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(get_compact_degree_lists_vertices, G, c, calc_until_layer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            degree_list.update(dl)

    logging.info("Saving degreeList on disk...")
    save_variable_on_disk(degree_list, 'compactDegreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1 - t0) / 60))

    return


def exec_bfs(G, workers, calcUntilLayer):
    futures = {}
    degree_list = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices, parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(get_degree_lists_vertices, G, c, calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            degree_list.update(dl)

    logging.info("Saving degreeList on disk...")
    save_variable_on_disk(degree_list, 'degreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1 - t0) / 60))

    return


def generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1, parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances = restore_variable_from_disk('distances-' + str(part))

        for vertices, layers in distances.iteritems():
            for layer, distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if layer not in weights_distances:
                    weights_distances[layer] = {}
                weights_distances[layer][vx, vy] = distance

        logging.info('Part {} executed.'.format(part))

    for layer, values in weights_distances.iteritems():
        save_variable_on_disk(values, 'weights_distances-layer-' + str(layer))
    return


def generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1, parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances = restore_variable_from_disk('distances-' + str(part))

        for vertices, layers in distances.iteritems():
            for layer, distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if layer not in graphs:
                    graphs[layer] = {}
                if vx not in graphs[layer]:
                    graphs[layer][vx] = []
                if vy not in graphs[layer]:
                    graphs[layer][vy] = []
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        logging.info('Part {} executed.'.format(part))

    for layer, values in graphs.iteritems():
        save_variable_on_disk(values, 'graphs-layer-' + str(layer))

    return


def generate_distances_network_part3():
    layer = 0
    while (is_pickle('graphs-layer-' + str(layer))):
        graphs = restore_variable_from_disk('graphs-layer-' + str(layer))
        weights_distances = restore_variable_from_disk('weights_distances-layer-' + str(layer))

        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v, neighbors in graphs.iteritems():
            e_list = deque()
            sum_w = 0.0

            for n in neighbors:
                if (v, n) in weights_distances:
                    wd = weights_distances[v, n]
                else:
                    wd = weights_distances[n, v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        save_variable_on_disk(weights, 'distances_nets_weights-layer-' + str(layer))
        save_variable_on_disk(alias_method_j, 'alias_method_j-layer-' + str(layer))
        save_variable_on_disk(alias_method_q, 'alias_method_q-layer-' + str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while (is_pickle('graphs-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restore_variable_from_disk('graphs-layer-' + str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving distancesNets on disk...")
    save_variable_on_disk(graphs_c, 'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return


def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while (is_pickle('alias_method_j-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = restore_variable_from_disk('alias_method_j-layer-' + str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    save_variable_on_disk(alias_method_j_c, 'nets_weights_alias_method_j')

    return


def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while (is_pickle('alias_method_q-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_q = restore_variable_from_disk('alias_method_q-layer-' + str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    save_variable_on_disk(alias_method_q_c, 'nets_weights_alias_method_q')

    return


def generate_distances_network(workers):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm " + return_path_struc2vec() + "/../pickles/weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1, workers)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm " + return_path_struc2vec() + "/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2, workers)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm " + return_path_struc2vec() + "/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm " + return_path_struc2vec() + "/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm " + return_path_struc2vec() + "/../pickles/alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info('- Time - part 6: {}s'.format(t))

    return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

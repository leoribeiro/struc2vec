# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,random,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from collections import defaultdict

from utils import *

epsilon = 0.01

def generate_parameters_random_walk():

    logging.info('Carregando distances_nets do disco...')

    graphs = restoreVariableFromDisk('distances_nets_graphs')
    weights = restoreVariableFromDisk('distances_nets_weights')

    sum_weights = {}
    amount_edges = {}
    for k,list_weights in weights.iteritems():
        layer = k[0]
        if(layer not in sum_weights):
            sum_weights[layer] = 0
        if(layer not in amount_edges):
            amount_edges[layer] = 0

        for w in list_weights:
            sum_weights[layer] += w
            amount_edges[layer] += 1

    average_weight = {}
    for layer in sum_weights.keys():
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]

    logging.info("Salvando average_weights no disco...")
    saveVariableOnDisk(average_weight,'average_weight')


    amount_neighbours = {}
    for k,list_weights in weights.iteritems():
        layer = k[0]
        cont_neighbours = 0
        for w in list_weights:
            if(w <= average_weight[layer]):
                cont_neighbours += 1
        amount_neighbours[k] = cont_neighbours

    logging.info("Salvando amount_neighbours no disco...")
    saveVariableOnDisk(amount_neighbours,'amount_neighbours')

def chooseNeighbor(v,graphs,weights,layer):
    v_list = graphs[layer,v]
    w_list = weights[layer,v]

    v_list = np.array(v_list,dtype='int')
    w_list = np.array(w_list,dtype='float')
    v = np.random.choice(v_list, p=w_list)

    return v


def exec_random_walk(graphs,weights,v,walk_length,diameter,amount_neighbours):
    #t0 = time()
    initialLayer = 0
    layer = initialLayer


    path = deque()
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        #### Navega pela camada
        if(r < 0.33):
                v = chooseNeighbor(v,graphs,weights,layer)
                path.append(v)
        #### Visita outras camadas
        else:
            r = random.random()
            limiar_moveup = prob_moveup(amount_neighbours[layer,v])
            #limiar_moveup = 0.5
            if(r > limiar_moveup):
                if(layer > initialLayer):
                    layer = layer - 1           
            else:
                if(layer < diameter):
                    if((layer + 1,v) in graphs):
                        layer = layer + 1

    #t1 = time()
    #logging.info('RW para vértice {} executada em : {}s'.format(v,(t1-t0)))

    return path


def exec_ramdom_walks_for_chunck(vertices,graphs,weights,walk_length,diameter,amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(graphs,weights,v,walk_length,diameter,amount_neighbours))

    return walks



def generate_random_walks(num_walks,walk_length,workers,diameter):

    graphs = Manager().dict()
    weights = Manager().dict()
    amount_neighbours = Manager().dict()

    logging.info('Carregando distances_nets do disco...')

    graphs_s = restoreVariableFromDisk('distances_nets_graphs')
    weights_s = restoreVariableFromDisk('distances_nets_weights')
    logging.info('Carregando amount_neighbours e average_weight do disco...')
    amount_neighbours_s = restoreVariableFromDisk('amount_neighbours')
    graphs.update(graphs_s)
    weights.update(weights_s)
    amount_neighbours.update(amount_neighbours_s)

    logging.info('Criando RWs...')
    t0 = time()
    
    walks = deque()
    initialLayer = 0

    vs = set()
    for k in graphs.keys():
        v = k[1]
        vs.add(v)

    vertices = list(vs)
    parts = workers

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for walk_iter in range(num_walks):
            futures = {}
            random.shuffle(vertices)
            chunks = partition(vertices,parts)
            part = 1
            for c in chunks:
                job = executor.submit(exec_ramdom_walks_for_chunck,c,graphs,weights,walk_length,diameter,amount_neighbours)
                futures[job] = part
                part += 1

            logging.info("Recebendo resultados...")
            for job in as_completed(futures):
                walk = job.result()
                r = futures[job]
                logging.info("Iteração: {} - RWs da parte {} executadas.".format(walk_iter,r))
                walks.extend(walk)
                del futures[job]

    t1 = time()
    logging.info('RWs criadas em : {}m'.format((t1-t0)/60))

    walks = list(walks)

    logging.info("Salvando Random Walks no disco...")
    saveVariableOnDisk(walks,'random_walks')


def exec_random_walk_version2(graphs,weights,walk_length,amount_neighbours,visits_node,diameter,num_vertices):
    t0 = time()
    initialLayer = 0
    layer = initialLayer

    visits_nodes = [0] * (num_vertices + 1)
    num_visited_vertices = 0

    vs = set()
    for k in graphs.keys():
        v = k[1]
        vs.add(v)
    vs = list(vs)
    ## Escolhe vértice inicial da RW
    v = random.choice(vs)

    path = deque()
    visits_nodes[v] += 1
    path.append(v)

    #while len(path) < walk_length and num_visited_vertices < num_vertices:
    while num_visited_vertices < num_vertices:
        r = random.random()

        #### Navega pela camada
        if(r < 0.33):
                v = chooseNeighbor(v,graphs,weights,layer)
                path.append(v)

                visits_nodes[v] += 1
                if(visits_nodes[v] == visits_node):
                    num_visited_vertices += 1
                
        #### Visita outras camadas
        else:
            r = random.random()
            limiar_moveup = prob_moveup(amount_neighbours[layer,v])
            if(r > limiar_moveup):
                if(layer > initialLayer):
                    layer = layer - 1           
            else:
                if(layer < diameter):
                    if((layer + 1,v) in graphs):
                        layer = layer + 1

    t1 = time()
    logging.info('Quantidade de vértices visitados {} vezes ou mais: {}'.format(visits_node,num_visited_vertices))
    logging.info('RW executada em : {}s'.format((t1-t0)))

    return path

def prob_moveup(amount_neighbours):
    #return 0.5
    p = (1.0 - (1.0 / (math.log(amount_neighbours + math.e) + epsilon) ))
    return p


def generate_random_walk(visits_node,diameter):

    logging.info('Carregando distances_nets do disco...')
    graphs = restoreVariableFromDisk('distances_nets_graphs')
    weights = restoreVariableFromDisk('distances_nets_weights')

    logging.info('Carregando amount_neighbours e average_weight do disco...')
    amount_neighbours = restoreVariableFromDisk('amount_neighbours')
    #average_weight = restoreVariableFromDisk('average_weight')

    vs = set()
    for k in graphs.keys():
        v = k[1]
        vs.add(v)

    num_vertices = len(vs)
    logging.info('Número de vértices: {}'.format(num_vertices))

    walk_length = 1000.0 * visits_node * num_vertices * math.log(float(num_vertices))
    logging.info('Tamanho máximo da RW: {}'.format(walk_length))

    walk = exec_random_walk_version2(graphs,weights,walk_length,amount_neighbours,visits_node,diameter,num_vertices)

    walks = [walk]

    logging.info('Tamanho da RW: {}'.format(len(walk)))

    logging.info("Salvando Random Walks no disco...")
    saveVariableOnDisk(walks,'random_walks')



#################################################################################################################

def removeVertices(d):
    vs = set()
    for v in d.keys():
        vs.add(v)
    for v,w in d.iteritems():
        n_w = list(set(w).intersection(vs))
        d[v] = n_w



def getBall(g, root, max_depth):
    t0 = time()
    #listas = []
    # cria vetor de marcação

    vetor_marcacao = [0] * (max(g.keys()) + 1)


    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)

    vetor_marcacao[root] = 1
    
    d = {}

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d[vertex] = list(g[vertex])

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1
  

        if(timeToDepthIncrease == 0):

            if depth == max_depth:
                removeVertices(d)
                return d

            depth += 1

            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('Tempo da BFS - vertice: {}: {}s'.format(root,(t1-t0)))

    removeVertices(d)
    return d


def simple_random_walk(g,start,path_length,alpha=0):

    path = [start]

    while len(path) < path_length:
      cur = path[-1]
      if len(g[cur]) > 0:
        if random.random() >= alpha:
          path.append(random.choice(g[cur]))
        else:
          path.append(path[0])
      else:
        break
    return path

def create_ball_and_random_walk(g,v,walk_length_balls,max_depth):
    logging.info('Criando ball do vértice {}...'.format(v))
    d = getBall(g, v, max_depth)
    logging.info('Ball criada.')
    walk = simple_random_walk(d,v,walk_length_balls)

    return walk


def generate_random_walks_balls(g,workers,walk_length_balls):

    g_s = Manager().dict()
    g_s.update(g)
    walks = []
    max_depth = 2

    with ProcessPoolExecutor(max_workers = workers) as executor:
        futures = {}

        for v in g.keys():
            job = executor.submit(create_ball_and_random_walk,g_s,v,walk_length_balls,max_depth)
            futures[job] = v

        for job in as_completed(futures):
            walk = job.result()
            walks.append(walk)
            logging.info('Recebendo RW do vértice: {}'.format(futures[job]))
            del futures[job]


    logging.info("Salvando Random Walks no disco...")
    saveVariableOnDisk(walks,'random_walks_balls')








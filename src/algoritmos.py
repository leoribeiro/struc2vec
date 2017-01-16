# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,random,logging
from fastdtw import fastdtw
import cPickle as pickle
import utils_networkx
import graph
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from collections import defaultdict

epsilon = 0.0000001

def getDegreeLists(g, root):
    t0 = time()

    listas = {}
    #listas = []
    # cria vetor de marcação
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    
    l = deque()
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.pop()
        timeToDepthIncrease -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='float')
            lp = np.sort(lp)
            #listas.append(lp)
            listas[depth] = lp


            l = deque()

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('Tempo da BFS - vertice: {}: {}s'.format(root,(t1-t0)))


    return listas

def calcDists(dm,dd):


    t0 = time()
    delta, path = fastdtw(dm,dd,radius=1,dist=custo)
    t1 = time()

    logging.info('FastDTW - Tempo: {}s . Distância: {}'.format(vm,vd,camada,(t1-t0),delta))


    return delta


def custo(a,b):
    m = max(a,b) + epsilon
    mi = min(a,b) + epsilon
    #print m,mi
    return math.log(m / mi)


def calc_distances_from_v(v,degreeList,calcUntilLayer):

    t00 = time()

    distances = {}

    lists = degreeList[v]
    size_l = len(lists)


    keys = [vd for vd in degreeList.keys() if vd > v]
    dList = { k : degreeList[k] for k in keys }


    for vd in dList.keys():

        lists_d = dList[vd]

        distances[v,vd] = {}

        maxLayer = min(size_l,len(lists_d),calcUntilLayer + 1)
        for layer in range(0,maxLayer):
            t0 = time()
            dist, path = fastdtw(lists[layer],lists_d[layer],radius=1,dist=custo)
            t1 = time()
            logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v,vd,layer,(t1-t0),dist))    

            distances[v,vd][layer] = dist


    t11 = time()
    logging.info('Tempo fastDTW de cálculo de todas as distâncias para o vértice {}: {}m'.format(v,(t11-t00)/60))    

    return distances



def calc_distances_with_list(vertices,degreeList,layer):

    distances = {}

    for vs in vertices:
        vm = vs[0]
        vd = vs[1]

        if(layer in degreeList[vm]) and (layer in degreeList[vd]):
            t0 = time()
            listvm = degreeList[vm][layer]
            listvd = degreeList[vd][layer]
            dist, path = fastdtw(listvm,listvd,radius=1,dist=custo)
            t1 = time()
            logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(vm,vd,layer,(t1-t0),dist))    

            distances[vm,vd] = dist
  

    return distances

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


def updateDistances(distances,new_distances,layer):

    for vs,dist in new_distances.iteritems():
        vm = vs[0]
        vd = vs[1]
        distances[vm,vd][layer] = dist

def restoreVariableFromDisk(name):
    logging.info('Recuperando variável...')
    t0 = time()
    val = None
    with open(name+'.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variável recuperada em {}m'.format((t1-t0)/60))

    return val

def saveVariableOnDisk(f,name):
    logging.info('Salvando variável no disco...')
    t0 = time()
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variável salva no disco em {}m'.format((t1-t0)/60))

    return

def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recuperando distâncias do disco...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecionando vértices...')

    vertices_selected = deque()

    for vertices,layers in distances.iteritems():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vértices selecionados.')

    return vertices_selected

def calcThresholdDistance(layer,distances,fractionCalcDists):

    logging.info('Calculando ThresholdDistance...')

    distances_layer = deque()

    for vertices,layers in distances.iteritems():
        if(layer not in layers):
            continue
        distances_layer.append(layers[layer])

    distances_layer = np.array(distances_layer,dtype='float')
    ids_distances = np.argwhere(distances_layer == -1.)
    distances_layer = np.delete(distances_layer,ids_distances)
    distances_layer = np.sort(distances_layer)

    fraction = int(len(distances_layer)*fractionCalcDists)
    distances_layer = distances_layer[0:fraction]

    if(len(distances_layer) > 1):
        threshold = distances_layer[-1]
    else:
        ## Chegou em uma camada que não possui vértices.
        threshold = -1

    logging.info('ThresholdDistance para passar da camada {} para a camada {} calculado: {}.'.format(layer,layer + 1,threshold))

    return threshold

def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidando distâncias...')

    for vertices,layers in distances.iteritems():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            #if(layers[layer] != -1.):
            layers[layer] += layers[layer - 1]

    logging.info('Distâncias consolidadas.')

def consolidesDistances(distances,layer):
        logging.info("Recuperando distâncias do disco...")
        total_distances = restoreVariableFromDisk('distances')
        updateDistances(total_distances,distances,layer)
        preprocess_consolides_distances(total_distances,layer)
        logging.info("Salvando distâncias no disco...")
        saveVariableOnDisk(total_distances,'distances')

def getDiameter(G):
    GX = utils_networkx.dictToGraph(G)
    logging.info('Número de vértices do grafo: {}'.format(GX.number_of_nodes()))
    logging.info('Número de arestas do grafo: {}'.format(GX.number_of_edges()))
    diameter = utils_networkx.getDiameter(GX)
    logging.info("Salvando diâmetro no disco...")
    saveVariableOnDisk(diameter,'diameter')

def exec_bfs(G,workers):

    futures = {}
    degreeList = {}

    t0 = time()

    with ProcessPoolExecutor(max_workers=workers) as executor:

        for v in G.keys():
            job = executor.submit(getDegreeLists,G,v)
            futures[job] = v

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList[v] = dl

    logging.info("Salvando degreeList no disco...")
    saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Tempo para a execução das BFS: {}m'.format((t1-t0)/60))


    return

def generate_distances_network(diameter):

        distances = restoreVariableFromDisk('distances')
        
        logging.info('Criando as redes de distâncias...')
        t0 = time()
        graphs = {}
        weights = {}
        for layer in range(0,diameter + 1):
            graphs[layer] = defaultdict(list)
            weights[layer] = defaultdict(float)


        for vertices,layers in distances.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
                weights[layer][vx,vy] = distance
                weights[layer][vy,vx] = distance
        
        logging.info('Redes de distâncias criadas com sucesso.')
        t1 = time()
        logging.info('Redes de distâncias criadas em : {}m'.format((t1-t0)/60))

        logging.info("Salvando distancesNets no disco...")
        saveVariableOnDisk(graphs,'distances_nets_graphs')
        saveVariableOnDisk(weights,'distances_nets_weights')

        
        
        return

def chooseNeighbor(v,g,weights):
    #t00 = time()

    e_list = deque()
    v_list = deque()

    sum_w = 0.0

    #t0 = time()
    for e in g[v]:
        w = 1.0 / (float(weights[v,e]) + epsilon)
        e_list.append(w)
        sum_w += w
        v_list.append(e)

    #t1 = time()
    #logging.info('Listas de pesos e vizinhos salvas em {}s'.format((t1-t0)))

    #t0 = time()
    if(sum_w != 0):
        e_list = [x / sum_w for x in e_list]
    else:
        e_list = [1.0 / len(e_list) for x in e_list]
    #t1 = time()
    #logging.info('Pesos normalizados em {}s'.format((t1-t0)))

    #t0 = time()
    v = np.random.choice(v_list, p=e_list)
    #t1 = time()
    #logging.info('Método random.choice executado em {}s'.format((t1-t0)))

    #t11 = time()
    #logging.info('Método escolhe vizinho executado em {}s'.format((t11-t00)))

    return v



def exec_random_walk(graphs,weights,v,walk_length):
    initialLayer = 0
    layer = initialLayer


    path = deque()
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        #### Navega pela camada
        if(r < 0.33):
                v = chooseNeighbor(v,graphs[layer],weights[layer])
                path.append(v)
        #### Visita outras camadas
        else:
            r = random.random()
            if(r < 0.5):
                if(layer > initialLayer):
                    layer = layer - 1           
            else:
                if(layer < len(graphs) - 1):
                    if(v in graphs[layer + 1]):
                        layer = layer + 1

    return path

def generate_random_walks(num_walks,walk_length,workers):

    initialLayer = 0
    graphs = restoreVariableFromDisk('distances_nets_graphs')
    weights = restoreVariableFromDisk('distances_nets_weights')

    logging.info('Criando RWs...')
    t0 = time()

    futures = {}
    walks = []
    nodes = list(graphs[initialLayer].keys())


    manager = Manager()
    graphs_s = manager.dict()
    graphs_s.update(graphs)
    weights_s = manager.dict()
    weights_s.update(weights)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                logging.info("Criando RW para o vértice {} - Iteração {}".format(node,walk_iter))
                job = executor.submit(exec_random_walk,graphs_s,weights_s,node,walk_length)
                futures[job] = (node,walk_iter)

        logging.info("Recebendo resultados...")
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            logging.info("RW executada para o vértice: {}  - Iteração: {}".format(r[0],r[1]))
            walks.append(walk)

    t1 = time()
    logging.info('RWs criadas em : {}m'.format((t1-t0)/60))

    logging.info("Salvando Random Walks no disco...")
    saveVariableOnDisk(walks,'random_walks')






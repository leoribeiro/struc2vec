# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from numpy import linalg as LA

from utils import *

epsilon = 0.01

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
        vertex = queue.popleft()
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

def custo(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep

    #print m,mi
    return ((m/mi) - 1)
    #return math.log(m / mi)


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
            #print dist
            #distances[v,vd][layer] = math.exp(dist)


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


def updateDistances(distances,new_distances,layer):

    for vs,dist in new_distances.iteritems():
        vm = vs[0]
        vd = vs[1]
        distances[vm,vd][layer] = dist


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

def preprocess_calculate_maxdistance():

    distances = restoreVariableFromDisk('distances')

    logging.info('Calculando exp das distâncias...')

    maxDistance = defaultdict(float)
    for vertices,layers in distances.iteritems():
        for layer,distance in layers.iteritems():
            if(distance > maxDistance[layer]):
                maxDistance[layer] = distance

    for vertices,layers in distances.iteritems():
        for layer,distance in layers.iteritems():
            r = np.exp(distance - maxDistance[layer])
            if(r == 0):
                print r, distance,maxDistance[layer]
            distances[vertices][layer] = r
            #distances[vertices][layer] = math.exp(distance)

    logging.info('Exp das distâncias calculado.')

    saveVariableOnDisk(distances,'distances')


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
        weights_distances = {}
        weights = {}
        for layer in range(0,diameter + 1):
            weights_distances[layer] = defaultdict(float)


        for vertices,layers in distances.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if((layer,vx) not in graphs):
                    graphs[layer,vx] = []
                if((layer,vy) not in graphs):
                    graphs[layer,vy] = []
                graphs[layer,vx].append(vy)
                graphs[layer,vy].append(vx)
                weights_distances[layer][vx,vy] = distance
                weights_distances[layer][vy,vx] = distance

        logging.info('Transformando distâncias em pesos...')
        
        for k,neighbors in graphs.iteritems():
            layer = k[0]
            v = k[1]
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                w = np.exp(-float(weights_distances[layer][v,n]))
                #print w,weights_distances[layer][v,n]
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[layer,v] = list(e_list)


        logging.info('Pesos criados com sucesso.')
        
        logging.info('Redes de distâncias criadas com sucesso.')
        t1 = time()
        logging.info('Redes de distâncias criadas em : {}m'.format((t1-t0)/60))

        logging.info("Salvando distancesNets no disco...")
        saveVariableOnDisk(graphs,'distances_nets_graphs')
        saveVariableOnDisk(weights,'distances_nets_weights')
        
        
        return


def calcSpectralGap():
    weights = restoreVariableFromDisk('distances_nets_weights')
    graphs = restoreVariableFromDisk('distances_nets_graphs')
    matrix = {}

    vertices = set()
    layers = set()
    for k,ws in weights.iteritems():
        layer = k[0]
        v = k[1]
        vertices.add(v)
        layers.add(layer)

    vertices = list(vertices)
    layers = list(layers)
    vertices = sorted(vertices)
    lenxy = len(vertices)

    print vertices

    for l in layers:
        matrix[l] = [[0.0 for x in range(lenxy)] for y in range(lenxy)]

    for l in layers:
        c = 0
        for x in range(lenxy):
            cont = 0
            v_x = vertices[x]
            for y in range(lenxy):
                if((l,v_x) in graphs):
                    if(vertices[y] in graphs[l,v_x]):
                        idx = graphs[l,v_x].index(vertices[y])
                        matrix[l][x][y] = weights[l,v_x][idx] 
                        cont += weights[l,v_x][idx] 
                        c += 1
        print c,l

            #print cont,vertices[x],l

    #print matrix[13]
    # for k,l in matrix.iteritems():
    #     for line in l:
    #         if(int(sum(line)) != 1):
    #             print int(sum(line)),"%.30f" % sum(line)

    #print len(matrix[0])
    #print conts
    #print (sum(matrix[0][0]))
    for k,m in matrix.iteritems():
        a = np.array(m)

        #print (sum(m[2]))

        w, v = LA.eig(a.transpose())
        w = sorted(w,reverse=True)
        #saveVariableOnDisk(w,'autovalores')
        print 'Camada:',k
        print w
        print '-----'




# -*- coding: utf-8 -*-
from time import time
import logging
import cPickle as pickle
import utils_networkx

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

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

def getDiameter(G):
    GX = utils_networkx.dictToGraph(G)
    ne = sum([len(G[x]) for x in G.keys()])/2
    logging.info('Número de arestas do grafo (dict): {} '.format(ne))
    logging.info('Número de vértices do grafo: {}'.format(GX.number_of_nodes()))
    logging.info('Número de arestas do grafo: {}'.format(GX.number_of_edges()))
    diameter = utils_networkx.getDiameter(GX)
    logging.info("Salvando diâmetro no disco...")
    saveVariableOnDisk(diameter,'diameter')


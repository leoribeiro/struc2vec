#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,csv,scipy,scipy.spatial,scipy.stats,logging
from collections import defaultdict
import numpy as np

#### local import
from utils import *

logging.basicConfig(filename='grid_search.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

edge_list = "../graph/karate.edgelist"
mapeamento = "mapeamento-vertices.pickle"


dimensions = [2,4,5,10,30,100]
walk_lengths = [3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120] 
num_walks = [1,3,5,10,15,20,25,30,40,50,60,70,80,90,100,200,400,500,1000]
window_sizes = [1,2,3,5,10,12,15,20]
pq = [0.25,0.50,1,2,4]

dimensions = [2]
walk_lengths = [3,5] 
num_walks = [1,3]
window_sizes = [1,2]



def exec_node2vec(d,w,n,z,o):

	os.system('python main.py --input ' + edge_list + ' --dimensions ' + str(d) +
		' --walk-length ' + str(w) + ' --num-walks ' + str(n) + ' --window-size ' + str(z)
		+ ' --p ' + str(p) + ' --q ' + str(q) + ' --output ' + o)

	# os.system('python plot_graph.py --edge-list ' + edge_list + ' --dict-map ' + mapeamento
	# 	+ ' --emb-file ' + o)


def read_file(o):
	X = {}
	with open(o,'r') as f:
	    reader = csv.reader(f, delimiter=' ')
	    for i, row in enumerate(reader):
	        index = int(row[0])
	        feats = map(float, row[1:])
	        X[index] = np.array(feats)
	return X

def calc_euclidian_dists(o):
	X = read_file(o)
	dists = {}
	for u in X.keys():
		for v in X.keys():
			if(u != v):
				dists[u,v] = scipy.spatial.distance.euclidean(X[u],X[v])

	return dists

def calc_network_dists():
	graphs = restoreVariableFromDisk('distances_nets_graphs')
	weights = restoreVariableFromDisk('distances_nets_weights')

	dists = {}

	for k,nbs in graphs.iteritems():
		layer = k[0]
		u = k[1]
		for idx,v in enumerate(nbs):
			if((u,v) not in dists):
				dists[u,v] = {}
			dists[u,v][layer] = weights[k][idx]

	dists_r = {}
	cont_r = {}
	for k,layers in dists.iteritems():
		u = k[0]
		v = k[1]

		if((u,v) not in dists_r and (v,u) not in dists_r):
				dists_r[u,v] = 0
				cont_r[u,v] = 0

		if((u,v) not in dists_r):
			temp = u
			u = v
			v = temp

		for l in layers.values():
			dists_r[u,v] += l
			cont_r[u,v] += 1

	for k in dists_r.keys():
		dists_r[k] = cont_r[k] / dists_r[k]

	return dists_r

def trataDistancias(d1,d2):
	v1 = []
	v2 = []
	for k in d1.keys():
		v1.append(d1[k])
		v2.append(d2[k])

	return v1,v2

def calc_correlation(d1,d2):
	sp = scipy.stats.spearmanr(d1,d2)
	pe = scipy.stats.pearsonr(d1,d2)
	logging.info("Spearmanr: {}".format(sp))
	logging.info("Pearsonr: {}".format(pe))
	logging.info("--")
	return sp,pe



################################################################




corr = 0
config = {}
for d in dimensions:
	for w in walk_lengths:
		for n in num_walks:
			for z in window_sizes:
				logging.info("Dimension: {} Walk-length: {} Num walks: {} Window size: {}.".format(d,w,n,z))
				o = 'karate-d'+str(d)+'-wl'+str(w)+'-nw'+str(n)+'-wz'+str(z)+'.emb'
				o = '../dados/' + o
				exec_struc2vec(d,w,n,z,o)
				dists_e = calc_euclidian_dists(o)
				dists_n = calc_network_dists()
				#print len(dists_e),len(dists_n)
				d1,d2 = trataDistancias(dists_n,dists_e)
				#print len(d1),len(d2)
				sp,pe = calc_correlation(d1,d2)
				if(abs(sp[0]) > corr):
					corr = sp[0]
					corr_v = sp
					config['dimension'] = d
					config['walk-length'] = w
					config['num-walks'] = n
					config['window-size'] = z



logging.info("Melhor correlação de Spearman: {}".format(corr_v))
logging.info("Configuração:")
logging.info("{}".format(config))




	    



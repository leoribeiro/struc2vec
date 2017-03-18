#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random
from collections import deque
import graph

def getEdgeList(g, root, size_net):

    edge_list = set()
    vertices = set()

    size = 0

    # cria vetor de marcação
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        for v in g[vertex]:

        	##############
            if((vertex,v) not in edge_list and (v,vertex) not in edge_list):
            	print vertex,len(g[vertex]),v,len(g[v])
            	edge_list.add((vertex,v))
            	vertices.add(vertex)
            	vertices.add(v)
            	if(len(vertices) >= size_net):
        			return edge_list
            ##############

            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1 

        if(timeToDepthIncrease == 0):

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    return edge_list

def generateNewFile(fileName,edge_list):

    n_file = open(fileName,'w')

    for e in edge_list:
        n_file.write(str(e[0])+" "+str(e[1])+"\n")

    n_file.close()


file_original_net = sys.argv[1]
size_net = int(sys.argv[2])

G = graph.load_edgelist(file_original_net,undirected=True)

root = random.choice(G.nodes())


edge_list = getEdgeList(G,root,size_net)

generateNewFile(file_original_net+"_amostra",edge_list)









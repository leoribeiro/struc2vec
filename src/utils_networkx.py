#!/usr/bin/env python
# -*- coding: utf-8 -*-

import graph, logging
import networkx as nx
#import graph_tool.all as gt

def dictToGraph(dictGraph):
  G = nx.from_dict_of_lists(dictGraph)
  return G

def getDiameter(G):
  logging.info("Calculando diâmetro...")

  if(nx.is_connected(G)):
    diameter = nx.diameter(G)
  else:
    #giant = max(nx.connected_component_subgraphs(G), key=len)
    subgrafos = nx.connected_component_subgraphs(G)
    diameter = 0
    for g in subgrafos:
      d = nx.diameter(g)
      if(d > diameter):
        diameter = d
  
  logging.info("Diâmetro calculado.")
  return diameter
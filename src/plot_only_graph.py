# -*- coding: utf-8 -*-

"""
Simple demo of a scatter plot.
"""
import numpy as np
import sys

import cPickle as pickle
import random
import argparse
from graph_tool.all import Graph,sfdp_layout,graph_draw
from gensim.models import Word2Vec


def getColorText(c):
	c = list(c)
	r = c[1] + c[2]
	r = float.fromhex(r)
	g = c[3] + c[4]
	g = float.fromhex(g)
	b = c[5] + c[6]
	b = float.fromhex(b)

	if(r*0.299 + g*0.587 + b*0.144) > 186:
		return "#000000"
	else:
		return "#ffffff"

def scatter_nodes(pos, labels=None, color=None, size=20, opacity=1):
    # pos is the dict of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None the Plotly default color is used
    # size is the size of the dots representing the nodes
    #opacity is a value between [0,1] defining the node color opacity
    L=len(pos[0])
    trace = Scatter(x=[], y=[],  mode='markers', marker=Marker(size=[]))
    for k in range(L):
        trace['x'].append(pos[0][k])
        trace['y'].append(pos[1][k])
    attrib=dict(name='', text=labels , hoverinfo='text', opacity=opacity) # a dict of Plotly node attributes
    trace=dict(trace, **attrib)# concatenate the dict trace and attrib
    trace['marker']['size']=size
    if color is not None:
    	trace['marker']['color']=color
    return trace  

def make_annotations(pos, text, colorVertex, font_size=14, font_color='rgb(25,25,25)'):
    L=len(pos[0])
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = Annotations()
    for k in range(L):
    	f = getColorText(colorVertex[k])
        annotations.append(
            Annotation(
                text=text[k], 
                x=pos[0][k], y=pos[1][k],
                xref='x1', yref='y1',
                font=dict(color= f, size=font_size),
                showarrow=False)
        )
    return annotations  


def trataCores(dictMap):

    colorsVertex = {}

    cont = 0
    for k,v in dictMap.iteritems():
    	r = lambda: random.randint(0,255)
    	t = ('#%02X%02X%02X' % (r(),r(),r()))
    	colorsVertex[k] = t
    	colorsVertex[v] = t
    	cont += 1

    return colorsVertex

def trataPosicoes(g,pos,dict_map,labels_vertices_inv):

	new_pos = g.new_vertex_property("vector<double>")

	for k,v in dict_map.iteritems():
		index = labels_vertices_inv[k]
		vertex_p = g.vertex(index)
		new_pos[vertex_p] = pos[vertex_p]

		index = labels_vertices_inv[v]
		vertex_s = g.vertex(index)
		
		n_pos = [pos[vertex_p][0]+20, pos[vertex_p][1]]
		new_pos[vertex_s] = n_pos		

	return new_pos

def mapeiaLabels(g,labels_vertices):
	dict_map_inv = {}
	for v in g.vertices():
		index = g.vertex_index[v]
		label = labels_vertices[v]
		dict_map_inv[label] = index
	return dict_map_inv		

parser = argparse.ArgumentParser(description='Espelhar grafo.')
parser.add_argument('--edge-list', nargs='?', required=True,
                      help='EdgeList a ser espelhada.')

args = parser.parse_args()


print "Carregando arquivo..."

g = Graph(directed=False)

edgelist = []

with open(args.edge_list) as f:
    for line in f:
    	if(line):
    		edgelist.append(map(int,line.split()))

labels_vertices = g.add_edge_list(edgelist,hashed=True)

labels_vertices_str = g.new_vertex_property("string")
for v in g.vertices():
	labels_vertices_str[v] = str(labels_vertices[v])

labels_vertices_inv = mapeiaLabels(g,labels_vertices)


color = g.new_vertex_property("string")



vprops = {}
vprops["font_size"] = 24
vprops["text"] = labels_vertices_str

#,vprops=vprops
#vcmap=cm.nipy_spectral,
graph_draw(g,output="grafo-.png",output_size=(2048,1024))

# graph_draw(g,vertex_fill_color=color,vertex_text=labels_vertices_str,vertex_font_size=20,
# 	pos=pos_new,output="grafo.png",output_size=(2048,1024))


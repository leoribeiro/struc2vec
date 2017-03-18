# -*- coding: utf-8 -*-

"""
Simple demo of a scatter plot.
"""
import numpy as np
import sys

import cPickle as pickle
import random
import argparse
from gensim.models import Word2Vec
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import networkx as nx

from plotly.plotly import image
from plotly.graph_objs import Layout, Font, XAxis, YAxis, Margin, Scatter, Marker, Data, Figure, Annotation, Annotations

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


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


def trataCores(v):

    age = g.node[v]['age'] 
        
    if(age not in colorsAge):
        r = lambda: random.randint(0,255)
        t = ('#%02X%02X%02X' % (r(),r(),r()))
        colorsAge[age] = t

    colorsVertex[v] = colorsAge[age]



parser = argparse.ArgumentParser(description='Espelhar grafo.')
parser.add_argument('--pickle', nargs='?', required=True,
                      help='Arquivo pickle.')
parser.add_argument('--emb-file', nargs='?', required=True,
                      help='Arquivo de embeddings.')
args = parser.parse_args()


print "Carregando arquivo..."

colorsVertex = {}
colorsAge = {}

g = nx.read_gpickle(args.pickle)

embeddings = Word2Vec.load_word2vec_format(args.emb_file, binary=False)



x = []
y = []
n = []
colors_graph = []
for v in g.vertices():
	l = str(v)
	coords = embeddings[l]
	x.append(coords[0])
	y.append(coords[1])
	n.append(l)
    cor = trataCores(v)
	colors_graph.append(cor)



width=2048
height=1024
axis=dict(showline=True, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=True,
          showticklabels=True,
          )
layout=Layout(title= '',  #
    font= Font(),
    #showlegend=False,
    autosize=True,
    width=width,
    height=height,
    xaxis=XAxis(axis),
    yaxis=YAxis(axis),
    margin=Margin(
        l=25,
        r=10,
        b=20,
        t=10,
        pad=0,
       
    ),
    hovermode='closest',
    #plot_bgcolor='#EFECEA', #set background color            
    )

trace = scatter_nodes([x,y],n,colors_graph)

data=Data([trace])

fig = Figure(data=data, layout=layout)

fig['layout'].update(annotations=make_annotations([x,y], n, colors_graph))

image.save_as(fig,args.emb_file+"-grafico.png",scale=3)



'''
g.vp.vertex_name[v]
g.vertex_index[v]
g.vertex(index)
'''







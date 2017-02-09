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
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

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

def scatter_nodes(pos, labels=None, color=None, opacity=1):
    # pos is the dict of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None the Plotly default color is used
    # size is the size of the dots representing the nodes
    #opacity is a value between [0,1] defining the node color opacity
    L=len(pos[0])
    trace = Scatter(x=[], y=[],  mode='markers', marker = dict(
        size = 22,
        #size = 15,
        line = dict(
            width = 0.5,
            color = 'rgb(0, 0, 0)'
        )
    ))
    for k in range(L):
        trace['x'].append(pos[0][k])
        trace['y'].append(pos[1][k])
    attrib=dict(name='', text=labels , hoverinfo='text', opacity=opacity) # a dict of Plotly node attributes
    trace=dict(trace, **attrib)# concatenate the dict trace and attrib
    if color is not None:
    	trace['marker']['color']=color
    return trace  

def make_annotations(pos, text, colorVertex, font_size=11, font_color='rgb(25,25,25)'):
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

    c = ['#9ac974','#b9371d','#66a6f2','#dcc156','#815faa','#dddfe0']
    #c = ['#7CEA61', '#2D6D78', '#CA83E6', '#707EE7', '#12CF6D', '#FF2CF4', '#E4F401', '#8DB1B0', '#28D7A5', '#4A0A40', '#7DC884', '#B3C3EF', '#115FBB', '#A4719F', '#7F7DE1', '#BC6512', '#4822FA', '#E9062D', '#CA96E4', '#BC61B3', '#6B5BE6', '#EF31D1', '#D46033', '#09F25E', '#70490F', '#56F6B4', '#2D7155', '#62BEC7', '#F627CE', '#08888B', '#F8400C', '#4F75C6', '#9505C6', '#418482', '#C071DE', '#A6D606', '#DD7F43', '#667902', '#693192', '#FF1F95', '#833568', '#0142F3', '#C1C401', '#F6FDE7', '#BAE2E8', '#D87B36', '#CCD13B', '#3455CC', '#663A63', '#D56A43', '#A0B3C4', '#DDBBCD', '#574769', '#9C26F9', '#EAF6E4', '#9CC232', '#287602', '#33A0F8', '#21D5A0', '#3430B1', '#482358', '#2CF532', '#A3FFEA', '#B0A3E4', '#20980B', '#2C9744', '#9DA8A5', '#2101C9']
    cont = 0
    for k,v in dictMap.iteritems():
    	#r = lambda: random.randint(0,255)
    	#t = ('#%02X%02X%02X' % (r(),r(),r()))
    	colorsVertex[k] = c[cont]
    	colorsVertex[v] = c[cont]
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


while True:

    parser = argparse.ArgumentParser(description='Espelhar grafo.')
    parser.add_argument('--edge-list', nargs='?', required=True,
                          help='EdgeList a ser espelhada.')
    parser.add_argument('--dict-map', nargs='?', required=True,
                          help='Arquivo para mapeamento dos pares.')
    parser.add_argument('--emb-file', nargs='?', required=True,
                          help='Arquivo de embeddings.')
    args = parser.parse_args()
    with open(args.dict_map, 'rb') as handle:
        dict_map = pickle.load(handle)

    print dict_map

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

    pos = sfdp_layout(g)
    colors = trataCores(dict_map)
    pos_new = trataPosicoes(g,pos,dict_map,labels_vertices_inv)

    color = g.new_vertex_property("string")

    for v in g.vertices():
    	index = g.vertex_index[v]
    	label = labels_vertices[v]
    	if label not in colors:
    		colors[label] = '#0c0cff'
    		pos_new[v] = pos[v]
    	color[v] = colors[label]


    vprops = {}
    vprops["font_size"] = 24
    vprops["text"] = labels_vertices_str
    vprops["fill_color"] = color

    # #vcmap=cm.nipy_spectral,
    #graph_draw(g,vprops=vprops,pos=pos_new,output=args.emb_file+"-grafo.png",output_size=(2048,1024))

    # graph_draw(g,vertex_fill_color=color,vertex_text=labels_vertices_str,vertex_font_size=20,
    # 	pos=pos_new,output="grafo.png",output_size=(2048,1024))


    #graphviz_draw(g,vcolor=color,vcmap=cm.viridis,vsize=0.3,vprops=vprops, pos=pos_new,output="grafo.png")

    #########################

    embeddings = Word2Vec.load_word2vec_format(args.emb_file, binary=False)

    x = []
    y = []
    n = []
    colors_graph = []
    for v in g.vertices():
    	l = str(labels_vertices[v])
    	coords = embeddings[l]
    	x.append(coords[0])
    	y.append(coords[1])
    	n.append(l)
    	colors_graph.append(colors[labels_vertices[v]])

    # fig, ax = plt.subplots()
    # ax.scatter(x, y,c=colors_graph, s=320,cmap='nipy_spectral')

    # for i, txt in enumerate(n):

    # 	xx = str(float(x[i]) - 0.05)
    # 	yy = str(float(y[i]) - 0.04)

    # 	ax.annotate(txt.zfill(2), (xx,yy),color="white")

    #plt.savefig('grafico.png', format='png', dpi=300)
    #plt.show()

    width=2048
    height=1024
    axis=dict(showline=True, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=True,
              showticklabels=True
              )
    layout=Layout(title= '',  #
        font= Font(size=20),
        #showlegend=False,
        autosize=True,
        width=width,
        height=height,
        xaxis=XAxis(axis),
        yaxis=YAxis(axis),
        margin=Margin(
            l=55,
            r=20,
            b=40,
            t=10,
            pad=0,
           
        ),
        hovermode='closest',
        #plot_bgcolor='#EFECEA', #set background color            
        )

    trace2= scatter_nodes([x,y],n,colors_graph)

    data=Data([trace2])

    fig = Figure(data=data, layout=layout)

    #fig['layout'].update(annotations=make_annotations([x,y], n, colors_graph))
    #offline.iplot(fig, filename='tst')

    image.save_as(fig,args.emb_file+"-grafico.png",scale=3)

    a = raw_input('Pressione uma tecla para continuar: ')
    if(a and int(a) == 0):
        break



'''
g.vp.vertex_name[v]
g.vertex_index[v]
g.vertex(index)
'''







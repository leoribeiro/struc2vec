# -*- coding: utf-8 -*-

"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

def verificaLista(v,l):
	for ll in l:
		if(set(v) == set(ll)):
			return True
	return False

print "Carregando arquivo..."
reload(sys)  
sys.setdefaultencoding('utf8')

f = open(sys.argv[1], 'r')
lines = f.readlines()

dotFile = file(sys.argv[2], "w")

dotFile.write("graph " + sys.argv[2] + " {\n");

print "Iniciando geração do grafo..."

ver = []
ver2 = []
for l in lines:
	v = l.split()
	if(len(v) == 1):
		ver2.append(v[0])
		continue
	v1 = v[0]
	iterv = iter(v)
	next(iterv)
	for v2 in iterv:
		ver.append([v1,v2])

for v in ver:
	dotFile.write("\"" +str(v[0]) + "\" -- \"" + str(v[1]) +"\";\n");

for v in ver2:
	dotFile.write("\"" +str(v) + "\";\n");

dotFile.write("}\n");
dotFile.close()
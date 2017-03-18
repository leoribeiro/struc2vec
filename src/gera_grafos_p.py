#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random

def generateNewFile(fileName):

	n_file = open(fileName,'w')

	with open(edgeList) as infile:
	    for line in infile:
	        r = random.random()
	        if(r <= p):
	        	n_file.write(line)

	n_file.close()


edgeList = sys.argv[1]
p = 0.1

generateNewFile("grafo1.edgelist")
generateNewFile("grafo2.edgelist")





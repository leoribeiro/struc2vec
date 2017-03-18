# -*- coding: utf-8 -*-
import re
from numpy import linalg as LA
import numpy as np

d = {}

with open("espelhado-pedaco.edgelist","r") as f:
    for line in f:
        l = line.split()
        v1 = int(l[0])
        v2 = int(l[1])
        if(v1 not in d):
            d[v1] = set()
        if(v2 not in d):
            d[v2] = set()  
        d[v1].add(v2)
        d[v2].add(v1)

lenxy = len(d)
m = [[0 for x in range(lenxy)] for y in range(lenxy)]

for k in d.keys():
    s = len(d[k])
    vvs = []
    for x in range(len(d)):
        x_a = x + 1
        if(x_a in d[k]):
            m[k-1][x] = float(1)/s
        else:
            m[k-1][x] = 0.0
    print sum(m[k-1])
for x in m:
    print sum(x)

a = np.array(m)

w, v = LA.eig(a.transpose())
print w
# -*- coding: utf-8 -*-
import re
from numpy import linalg as LA
import numpy as np
import collections

d = {}

with open("barbell2.edgelist","r") as f:
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
m = [[0.0 for x in range(lenxy)] for y in range(lenxy)]

vertices = sorted(d.keys())

for x in range(lenxy):
    v_x = vertices[x]
    for y in range(lenxy):
        v_y = vertices[y]
        if(v_y in d[v_x]):
            s = len(d[v_x])
            m[x][y] = float(1)/s

for x in m:
    print sum(x)

a = np.array(m)

w, v = LA.eig(a.transpose())
print w
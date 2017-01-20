import numpy as np
import scipy.io
import sys

mat = scipy.io.loadmat(sys.argv[1])

adjMatrix = np.array(mat['network'].todense())
with open(sys.argv[2], 'w') as outfile:
    for i, firstVertexAdj in enumerate(adjMatrix,1):
        for j, edgeWeight in enumerate(firstVertexAdj,1):
            if(edgeWeight != 0): outfile.write("%s %s\n"%(i,j))

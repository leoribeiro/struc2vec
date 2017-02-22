# struc2vec

The *struc2vec* algorithm learns continuous representations for nodes in any graph that captures the structural equivalence between nodes.  

### Basic Usage

#### Example
To run *struc2vec* on Mirrored Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/karate-mirrored.edgelist --output emb/karate.emd``

#### Options
You can check out the other options available to use with *struc2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *struc2vec*.


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <leo@land.ufrj.br>.

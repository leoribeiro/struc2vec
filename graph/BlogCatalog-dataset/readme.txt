Social Computing Data Repository - Basic Information
==========================================================================
Dataset Name: BlogCatalog
Abstract: BlogCatalog is the social blog directory which manages the bloggers and their blogs.
Number of Nodes: 10,312
Number of Edges: 333,983
Number of Groups: 39
Missing Values: No

Source:
==========================================================================
Lei Tang*, Huan Liu*

* School of Computing, Informatics and Decision Systems Engineering, Arizona State University. E-mail: l.tang@asu.edu, huan.liu@asu.edu

Data Set Information:
==========================================================================
[I]. Brief description
This is the data set crawled from BlogCatalog ( http://www.blogcatalog.com ). BlogCatalog is a social blog directory website. 
This contains the friendship network crawled and group memberships. For easier understanding, all the contents are organized in CSV file format.

[II]. Basic statistics
Number of bloggers : 10,312
Number of friendship pairs: 333,983
Number of groups: 39

[III]. The data format

4 files are included:

1. nodes.csv
-- it's the file of all the users. This file works as a dictionary of all the users in this data set. It's useful for fast reference. It contains
all the node ids used in the dataset

2. groups.csv
-- it's the file of all the groups. It contains all the group ids used in the dataset

3. edges.csv
-- this is the friendship network among the bloggers. The blogger's friends are represented using edges. 
Since the network is symmetric, each edge is represented only once. Here is an example. 

1,2

This means blogger with id "1" is friend with blogger id "2".

4. group-edges.csv
-- the user-group membership. In each line, the first entry represents user, and the 2nd entry is the group index. 

If you need to know more details, please check the relevant papers and code:
http://www.public.asu.edu/~ltang9/social_dimension.html

Relevant Papers:
==========================================================================

1. Lei Tang and Huan Liu. Relational Learning via Latent Social Dimensions. In Proceedings of The 15th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD’09), Pages 817–826, 2009.

2. Lei Tang and Huan Liu. Scalable Learning of Collective Behavior based on Sparse Social Dimensions. In Proceedings of the 18th ACM Conference on Information and Knowledge Management (CIKM’09), 2009.

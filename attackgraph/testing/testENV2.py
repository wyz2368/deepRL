from attackgraph import DagGenerator as dag
import numpy as np
import random

# random.seed(2)
# np.random.seed(2)
# env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)
# env.randomDAG()
#
# print(env.G.nodes.data())
# print(env.G.edges)

env = dag.env_rand_gen_and_save("test_env")
print(env.G.nodes.data())
print(env.G.edges)
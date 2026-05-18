import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import matplotlib.pyplot as plt
from tribal import BaseTree
from tribal.expansion_graph import ConstructGraph
from tribal.lineage_tree import MPTR as MPTR_original
# from tribal.draw_tree import plot_tree
from tribal.mptr_dag import MPTR_DAG


# Hard coded sanity check

T = nx.DiGraph()
T.add_edges_from([
    ('r', 'f'),
    ('f', 'b'),
    ('f', 'c'),
    ('f', 'd'),
    ('f', 'e')
])
root = 'r'

# Key: Vertex Name, Value: Isotype (number it comes in the ordering 0, ..., n-1 for n isotypes)
isotype_labels = {
    'r': 0,  # root
    'b': 1,
    'c': 2,
    'd': 1,
    'e': 3
}

P = np.array([
    [0.6, 0.2, 0.1, 0.1],
    [0.0, 0.6, 0.3, 0.1],
    [0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 1.0]
])
cost = -np.log(P + 1e-9)

base_tree = BaseTree(T, root=root, id=0, name="hardcoded")

cg = ConstructGraph(cost, isotype_labels, root_identifier=root)
fg = cg.build(base_tree)


# Run the ILPs

mptr1 = MPTR_original(
    fg.G,
    base_tree.T,
    fg.find_terminals(),
    fg.iso_weights,
    fg.tree_to_graph,
    root=root,
)
score1, tree1 = mptr1.run()
print(f"Original MPTR objective = {score1:.4f}")

mptr2 = MPTR_DAG(
    fg.G,
    base_tree.T,
    fg.find_terminals(),
    fg.iso_weights,
    fg.tree_to_graph,
    root=root,
)
score2, tree2 = mptr2.run()
print(f"New MPTR_DAG objective = {score2:.4f}")

# Tree visualization (temporary, need to figure out why TRIBAL draw_tree has error but this works for now)

def draw_tree(tree,i):
    pos = graphviz_layout(tree,prog='dot',args=f"-Grankdir-TB")
    nx.draw(tree,pos,with_labels=True,node_size=500,font_size=8,arrows=True,arrowstyle="->",arrowsize=10,ax=axes[i])


fig,axes = plt.subplots(1, 3, figsize=(15,5))

draw_tree(fg.G,0)
axes[0].set_title("Expansion Graph")
draw_tree(tree1,1)
axes[1].set_title("Orginal ILP")
draw_tree(tree2,2)
axes[2].set_title("DAG ILP")

plt.tight_layout()
plt.show()
import networkx as nx
import numpy as np
from itertools import combinations
from dataclasses import dataclass
from tribal.expansion_graph import FlowGraph  # Pulls the dataclass from the original package

def name_node(tree, node, label, is_poly=False, is_leaf=False): 
    lab = str(node) + "_" + str(label)
    if is_poly:
         lab += "_p"
    return lab

def decode_node(node, is_leaf=False):
    codes = node.split("_")
    if is_leaf:
         name = codes[0]
    else:
         name = node
    if codes[-1] == "p":
         isotype = codes[-2]
    else:
         isotype = codes[-1]
    return int(isotype), name  

class ConstructGraphMulti:
    def __init__(self, states, characters, root_identifier="root") -> None:
        self.Graphs = []
        self.states = states
        self.characters = characters
        self.root_identifier = root_identifier

    def build(self, LinTree):
        """
        Builds the expansion graph for multi-character systems and 
        returns a FlowGraph object mimicking TRIBAL's structure.
        """
        tree = LinTree.T
        root_node = LinTree.root
        tree_id = LinTree.id
        states = self.states
        characters = self.characters
        
        G = nx.DiGraph()
        char_list = list(characters.keys())
        char_states = {c: list(range(len(characters[c].P))) for c in char_list}
        
        root_state = tuple(states[root_node][c] for c in char_list)
        root_id = f"{root_node}_{root_state}"
        G.add_node(root_id, orig=root_node, state=root_state)
        
        expanded_map = {n: [] for n in tree.nodes()}
        expanded_map[root_node].append(root_id)

        for u in nx.topological_sort(tree):
            if tree.out_degree(u) > 0:
                frontier = list(expanded_map[u])
                seen_in_u = set(frontier)
                
                while frontier:
                    curr_u_exp = frontier.pop(0)
                    curr_state = G.nodes[curr_u_exp]['state']
                    
                    for i, c in enumerate(char_list):
                        curr_val = curr_state[i]
                        for next_val in char_states[c]:
                            if next_val == curr_val: continue
                            prob = characters[c].P[curr_val][next_val]
                            
                            if prob > 0 and prob != np.inf:
                                new_state = list(curr_state)
                                new_state[i] = next_val
                                v_state = tuple(new_state)
                                v_exp = f"{u}_{v_state}"
                                
                                if v_exp not in seen_in_u:
                                    G.add_node(v_exp, orig=u, state=v_state)
                                    expanded_map[u].append(v_exp)
                                    seen_in_u.add(v_exp)
                                    frontier.append(v_exp)
                                
                                weight = -np.log(prob) if prob < 1.0 else 0.0
                                G.add_edge(curr_u_exp, v_exp, weight=weight)

            for v in tree.successors(u):
                is_leaf = (tree.out_degree(v) == 0)
                target_leaf_state = tuple(states[v][c] for c in char_list) if is_leaf else None

                for u_exp in expanded_map[u]:
                    u_state = G.nodes[u_exp]['state']
                    
                    if not is_leaf or u_state == target_leaf_state:
                        v_id = f"{v}_{u_state}"
                        if v_id not in G:
                            G.add_node(v_id, orig=v, state=u_state)
                            expanded_map[v].append(v_id)
                        G.add_edge(u_exp, v_id, weight=0.0)

                    for i, c in enumerate(char_list):
                        curr_val = u_state[i]
                        for next_val in char_states[c]:
                            if next_val == curr_val: continue
                            prob = characters[c].P[curr_val][next_val]
                            
                            if prob > 0 and prob != np.inf:
                                mutated_state = list(u_state)
                                mutated_state[i] = next_val
                                mutated_state = tuple(mutated_state)
                                
                                if is_leaf and mutated_state != target_leaf_state:
                                    continue
                                    
                                v_id = f"{v}_{mutated_state}"
                                if v_id not in G:
                                    G.add_node(v_id, orig=v, state=mutated_state)
                                    expanded_map[v].append(v_id)
                                    
                                weight = -np.log(prob) if prob < 1.0 else 0.0
                                G.add_edge(u_exp, v_id, weight=weight)

        leaves_in_G = [n for n in G.nodes() if tree.out_degree(G.nodes[n]['orig']) == 0]
        reachable_to_leaf = set()
        for leaf in leaves_in_G:
            reachable_to_leaf.add(leaf)
            reachable_to_leaf.update(nx.ancestors(G, leaf))

        expansion_graph = G.subgraph(reachable_to_leaf).copy()

        seq_weights = {}
        multi_weights = {}
        node_states = {}
        node_mapping = {}
        tree_to_graph = {n: [] for n in tree.nodes()}
        node_out_degree = {n: tree.out_degree[n] for n in tree.nodes()}

        for n in expansion_graph.nodes():
            orig_node = expansion_graph.nodes[n]['orig']
            state_tuple = expansion_graph.nodes[n]['state']
            
            node_states[n] = state_tuple
            node_mapping[n] = orig_node
            tree_to_graph[orig_node].append(n)

        for u, v, data in expansion_graph.edges(data=True):
            seq_weights[(u, v)] = 0  
            multi_weights[(u, v)] = data.get('weight', 0.0)

        fg = FlowGraph(
            id=tree_id,
            G=expansion_graph,
            seq_weights=seq_weights,
            iso_weights=multi_weights,
            isotypes=node_states,
            node_mapping=node_mapping,
            tree_to_graph=tree_to_graph,
            node_out_degree=node_out_degree
        )

        self.Graphs.append(fg)
        return fg
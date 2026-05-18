import networkx as nx
import random
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def random_binary_tree(n_leaves: int, seed: Optional[int] = None) -> nx.Graph:
    if n_leaves < 2:
        raise ValueError("Need at least 2 leaves for a binary tree.")

    rng = random.Random(seed)

    T = nx.Graph()
    T.add_edge(0, 1)
    next_node = 2

    for _ in range(n_leaves - 2):
        u, v = rng.choice(list(T.edges()))

        w = next_node
        next_node += 1

        T.remove_edge(u, v)
        T.add_edge(u, w)
        T.add_edge(w, v)

        x = next_node
        next_node += 1
        T.add_edge(w, x)

    return T

def random_rooted_binary_tree(
    n_leaves: int,
    seed: Optional[int] = None
) -> tuple[nx.DiGraph, int]:
    rng = random.Random(seed)

    T = random_binary_tree(n_leaves, seed=seed)

    u, v = rng.choice(list(T.edges()))

    r = max(T.nodes()) + 1 
    T.remove_edge(u, v)
    T.add_edge(u, r)
    T.add_edge(r, v)

    Di = nx.DiGraph()
    Di.add_nodes_from(T.nodes())
    Di.add_edges_from(nx.bfs_edges(T, source=r))

    return Di, r

@dataclass
class DiscreteCharacter:
    name: str
    n_states: int
    P: np.ndarray
    preserve_triangle: bool = False

    def sample_next_state(
        self,
        current_state: int,
        rng: Optional[np.random.Generator] = None
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        probs = self.P[current_state]
        return int(rng.choice(self.n_states, p=probs))
    
def threshold_and_renormalize(probs: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    cleaned_probs = np.where(probs < threshold, 0.0, probs)
    total = np.sum(cleaned_probs)
    
    if total > 0:
        return cleaned_probs / total
    else:
        fallback = np.zeros_like(probs)
        fallback[np.argmax(probs)] = 1.0
        return fallback

def random_transition_matrix(
    n_states: int, 
    alpha: float = 1.0,
    preserve_triangle: bool = False,
    max_iters: int = 10000
) -> np.ndarray:
    alphas = np.full(n_states, alpha, dtype=float)
    
    if not preserve_triangle:
        rows = [threshold_and_renormalize(np.random.dirichlet(alphas)) for _ in range(n_states)]
        return np.vstack(rows)
        
    for _ in range(max_iters):
        P = np.vstack([threshold_and_renormalize(np.random.dirichlet(alphas)) for _ in range(n_states)])
        
        # 2. Enforce Triangle Inequality using Metric Closure (Floyd-Warshall)
        # Convert probabilities to distance/cost (add epsilon to avoid log(0))
        eps = 1e-12
        cost_matrix = -np.log(np.clip(P, eps, 1.0))
        
        # Floyd-Warshall Algorithm: if a path through 'k' is cheaper, use it
        for k in range(n_states):
            for i in range(n_states):
                for j in range(n_states):
                    if cost_matrix[i, j] > cost_matrix[i, k] + cost_matrix[k, j]:
                        cost_matrix[i, j] = cost_matrix[i, k] + cost_matrix[k, j]
                        
        # 3. Convert back to probabilities
        P_fixed = np.exp(-cost_matrix)
        
        # 4. Renormalize the rows so they perfectly sum to 1.0 again
        for i in range(n_states):
            P_fixed[i] = threshold_and_renormalize(P_fixed[i])
        
        
        all_valid = True
        for i in range(n_states):
            for j in range(n_states):
                if i == j: continue
                for k in range(n_states):
                    if k == i or k == j: continue
                    if P_fixed[i, k] < P_fixed[i, j] * P_fixed[j, k]:
                        all_valid = False
                        break
                if not all_valid: break
            if not all_valid: break
            
        if all_valid:
            # print("Returned triangle ineq for fully connected transition matrix")
            return P_fixed
            
    print("Reached max iterations for fully connected transition matrix")
    return P_fixed

def random_irreversible_transition_matrix(
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    preserve_triangle: bool = False,
    max_iters: int = 10000
) -> np.ndarray:
    P = np.zeros((n_states, n_states), dtype=float)
    
    # Process bottom-up (reverse order) so descendants are available for checking
    for i in range(n_states - 1, -1, -1):
        k_len = n_states - i
        alphas = np.full(k_len, alpha, dtype=float)
        alphas[0] *= self_bias
        
        if not preserve_triangle:
            probs = threshold_and_renormalize(np.random.dirichlet(alphas))
            P[i, i:] = probs
        else:
            accepted = False
            for _ in range(max_iters):
                probs = threshold_and_renormalize(np.random.dirichlet(alphas))
                
                all_valid = True
                for j_idx in range(1, k_len):
                    j = i + j_idx
                    p_i_j = probs[j_idx]
                    
                    for k_idx in range(j_idx + 1, k_len):
                        k = i + k_idx
                        p_i_k = probs[k_idx]
                        
                        if p_i_k < p_i_j * P[j, k]:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                        
                if all_valid:
                    P[i, i:] = probs
                    accepted = True
                    break
                    
            if not accepted:
                print(f"Reached max iterations for irreversible matrix state {i}")
                P[i, i:] = probs
                
    return P

def make_reversible_character(
    name: str,
    n_states: int,
    alpha: float = 1.0,
    preserve_triangle: bool = False,
    max_iters: int = 10000
) -> DiscreteCharacter:
    P = random_transition_matrix(
        n_states, alpha=alpha, preserve_triangle=preserve_triangle, max_iters=max_iters
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P, preserve_triangle=preserve_triangle)


def make_irreversible_character(
    name: str,
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    preserve_triangle: bool = False,
    max_iters: int = 10000
) -> DiscreteCharacter:
    P = random_irreversible_transition_matrix(
        n_states=n_states, alpha=alpha, self_bias=self_bias, 
        preserve_triangle=preserve_triangle, max_iters=max_iters
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P, preserve_triangle=preserve_triangle)

def random_tree_transition_matrix_rejection(
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    root_state: int = 0,
    closure_prob: float = 0.0,
    preserve_triangle: bool = True,
    max_iters: int = 10000
) -> np.ndarray:
    if n_states < 1:
        print("Must have at least one states")
    if n_states == 1:
        return np.array([[1.0]])
        
    seq = list(np.random.randint(0, n_states, size=n_states - 2))
    undirected_tree = nx.from_prufer_sequence(seq)
    directed_edges = list(nx.bfs_edges(undirected_tree, source=root_state))
    
    T = nx.DiGraph()
    T.add_nodes_from(range(n_states))
    T.add_edges_from(directed_edges)
    
    P = np.zeros((n_states, n_states), dtype=float)
    
    nodes_bottom_up = list(reversed(list(nx.topological_sort(T))))
    
    for i in nodes_bottom_up:
        direct_children = list(T.successors(i))
        reachable = [i] + direct_children
        
        active_closures = []
        if closure_prob > 0:
            descendants = nx.descendants(T, i)
            closure_candidates = descendants - set(direct_children)
            for k in closure_candidates:
                if np.random.rand() < closure_prob:
                    reachable.append(k)
                    active_closures.append(k)
                    
        k_len = len(reachable)
        alphas = np.full(k_len, alpha, dtype=float)
        alphas[0] *= self_bias
        
        if not preserve_triangle or not active_closures:
            probs = threshold_and_renormalize(np.random.dirichlet(alphas))
            for idx, target_state in enumerate(reachable):
                P[i, target_state] = probs[idx]
        else:
            accepted = False
            for iter in range(max_iters):
                probs = threshold_and_renormalize(np.random.dirichlet(alphas))
                
                proposed_P = {target_state: probs[idx] for idx, target_state in enumerate(reachable)}
                
                all_valid = True
                for k in active_closures:
                    path = nx.shortest_path(T, source=i, target=k)
                    
                    prob_prod = 1.0
                    
                    prob_prod *= proposed_P[path[1]]
                    
                    for step in range(1, len(path) - 1):
                        u_step = path[step]
                        v_step = path[step+1]
                        prob_prod *= P[u_step, v_step]
                        
                    if proposed_P[k] < prob_prod:
                        all_valid = False
                        break
                        
                if all_valid:
                    for target_state, p_val in proposed_P.items():
                        P[i, target_state] = p_val
                    accepted = True
                    break
                    
            if not accepted:
                print("Reached max iterations")
                
    return P

def random_dag_transition_matrix(
    n_states: int,
    edge_prob: float = 0.5,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    preserve_triangle: bool = False,
    max_iters: int = 10000
) -> np.ndarray:
    P = np.zeros((n_states, n_states), dtype=float)
    
    if n_states == 1:
        P[0, 0] = 1.0
        return P

    reachable = {i: [i] for i in range(n_states)}
    for j in range(1, n_states):
        i = np.random.randint(0, j)
        reachable[i].append(j)
        
    for i in range(n_states):
        for j in range(i + 1, n_states):
            if j not in reachable[i]:
                if np.random.rand() < edge_prob:
                    reachable[i].append(j)
                    
    # Process bottom-up (reverse topological order)
    for i in range(n_states - 1, -1, -1):
        reach = sorted(reachable[i]) 
        k_len = len(reach)
        alphas = np.full(k_len, alpha, dtype=float)
        alphas[0] *= self_bias 
        
        if not preserve_triangle:
            probs = threshold_and_renormalize(np.random.dirichlet(alphas))
            for idx, target_state in enumerate(reach):
                P[i, target_state] = probs[idx]
        else:
            accepted = False
            for _ in range(max_iters):
                probs = threshold_and_renormalize(np.random.dirichlet(alphas))
                proposed_P = {target_state: probs[idx] for idx, target_state in enumerate(reach)}
                
                all_valid = True
                for j in reach:
                    if j == i: continue
                    # ensure transitioning through j doesn't override the direct path
                    for k in reachable[j]:
                        if k in proposed_P:
                            if proposed_P[k] < proposed_P[j] * P[j, k]:
                                all_valid = False
                                break
                    if not all_valid:
                        break
                        
                if all_valid:
                    for target_state, p_val in proposed_P.items():
                        P[i, target_state] = p_val
                    accepted = True
                    break
                    
            if not accepted:
                print(f"Reached max iterations for DAG matrix state {i}")
                for target_state, p_val in proposed_P.items():
                    P[i, target_state] = p_val
            
    return P

def make_dag_character(
    name: str,
    n_states: int,
    edge_prob: float = 0.5,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    preserve_triangle: bool = False
) -> DiscreteCharacter:
    P = random_dag_transition_matrix(
        n_states=n_states, 
        edge_prob=edge_prob, 
        alpha=alpha, 
        self_bias=self_bias,
        preserve_triangle=preserve_triangle
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P, preserve_triangle=preserve_triangle)

def make_tree_character(
    name: str,
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0,
    root_state: int = 0,
    closure_prob: float = 0.0,
    preserve_triangle: bool = True,
    max_iters: int = 10000
) -> DiscreteCharacter:
    P = random_tree_transition_matrix_rejection(
        n_states=n_states, 
        alpha=alpha, 
        self_bias=self_bias,
        root_state=root_state,
        closure_prob=closure_prob,
        preserve_triangle=preserve_triangle,
        max_iters=max_iters
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P)

def simulate_on_tree(
    tree: nx.DiGraph,
    root: int,
    characters: Dict[str, DiscreteCharacter],
    rng: Optional[np.random.Generator] = None
) -> Dict[int, Dict[str, int]]:

    if rng is None:
        rng = np.random.default_rng()

    states: Dict[int, Dict[str, int]] = {}

    root_state: Dict[str, int] = {}
    for name, char in characters.items():
        root_state[name] = 0
    states[root] = root_state

    for node in nx.topological_sort(tree):
        if node == root:
            continue

        parent = next(tree.predecessors(node))
        parent_state = states[parent]

        node_state: Dict[str, int] = {}
        for name, char in characters.items():
            parent_val = parent_state[name]
            node_state[name] = char.sample_next_state(parent_val, rng=rng)

        states[node] = node_state

    for node, st in states.items():
        for name, val in st.items():
            tree.nodes[node][name] = val

    return tree,states

def only_contracted_chars_changed(u, v, states, chars_to_contract):
    
    first_node = next(iter(states))
    all_chars = list(states[first_node].keys())
    other_chars = [c for c in all_chars if c not in chars_to_contract]

    for c in other_chars:
        if states[u][c] != states[v][c]:
            return False
    
    return True

def contract_tree_by_characters(tree, states, characters, chars_to_contract):
    tree = tree.copy()
    contracted_states = states.copy()

    contracted_round = True

    while(contracted_round):
        contracted_round = False
        for u,v in tree.edges():
            if tree.out_degree(v)==0:
                continue
            if only_contracted_chars_changed(u, v, states, chars_to_contract):
                for n in tree.successors(v):
                    tree.add_edge(u,n)
                tree.remove_edge(u,v)
                tree.remove_node(v)
                contracted_states.pop(v)
                contracted_round=True
                break

    filtered_states = {
        node: {char: val for char, val in char_dict.items() if char in chars_to_contract}
        for node, char_dict in contracted_states.items()
    }

    contracted_characters = {
        char: obj for char, obj in characters.items() if char in chars_to_contract
    }
    
    return tree, filtered_states, contracted_characters

def construct_expansion_graph(tree, states, characters, root_node):
    G = nx.DiGraph()
    char_list = list(characters.keys())
    char_states = {c: list(range(len(characters[c].P))) for c in char_list}
    
    root_state = tuple(states[root_node][c] for c in char_list)
    root_id = f"{root_node}_{root_state}"
    G.add_node(root_id, orig=root_node, state=root_state)
    
    expanded_map = {n: [] for n in tree.nodes()}
    expanded_map[root_node].append(root_id)

    for u in nx.topological_sort(tree):
        #u != root_node and
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

    return expansion_graph


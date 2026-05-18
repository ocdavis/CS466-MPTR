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

def random_transition_matrix(n_states: int, alpha: float = 1.0) -> np.ndarray:
    alphas = np.full(n_states, alpha, dtype=float)
    rows = [threshold_and_renormalize(np.random.dirichlet(alphas)) for _ in range(n_states)]
    return np.vstack(rows)


def random_irreversible_transition_matrix(
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0
) -> np.ndarray:
    P = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        k = n_states - i
        alphas = np.full(k, alpha, dtype=float)
        alphas[0] *= self_bias
        probs = threshold_and_renormalize(np.random.dirichlet(alphas))
        P[i, i:] = probs
    return P

def make_reversible_character(
    name: str,
    n_states: int,
    alpha: float = 1.0
) -> DiscreteCharacter:
    P = random_transition_matrix(n_states, alpha=alpha)
    return DiscreteCharacter(name=name, n_states=n_states, P=P)


def make_irreversible_character(
    name: str,
    n_states: int,
    alpha: float = 1.0,
    self_bias: float = 1.0
) -> DiscreteCharacter:
    P = random_irreversible_transition_matrix(
        n_states=n_states, alpha=alpha, self_bias=self_bias
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P)

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
    self_bias: float = 1.0
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
                    
    for i in range(n_states):
        reach = sorted(reachable[i]) 
        
        k = len(reach)
        alphas = np.full(k, alpha, dtype=float)
        alphas[0] *= self_bias 
        
        probs = threshold_and_renormalize(np.random.dirichlet(alphas))
        
        for idx, target_state in enumerate(reach):
            P[i, target_state] = probs[idx]
            
    return P

def make_dag_character(
    name: str,
    n_states: int,
    edge_prob: float = 0.5,
    alpha: float = 1.0,
    self_bias: float = 1.0
) -> DiscreteCharacter:
    P = random_dag_transition_matrix(
        n_states=n_states, 
        edge_prob=edge_prob, 
        alpha=alpha, 
        self_bias=self_bias
    )
    return DiscreteCharacter(name=name, n_states=n_states, P=P)

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
import networkx as nx 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import time

class MPTR_CP_GUR:
    def __init__(self, G, T, S, edge_weight, tree_to_graph, root=0, threads=1, timeout=600.0) -> None:
        self.G = G
        self.T = T
        self.terminals = S 
        self.tree_to_graph = tree_to_graph
        self.orig_root = root
        self.root = self.tree_to_graph[root][0]
        self.nodes = set(G.nodes)
        self.edges = set(G.edges)
        
        self.c = {e: edge_weight[e] for e in self.edges}
        self.threads = threads

    def createModel(self):
        self.m = gp.Model("MPTR_CP")
        self.m.Params.OutputFlag = 0
        self.m.Params.LazyConstraints = 1
        
        self.x = self.m.addVars(self.edges, vtype=GRB.BINARY, obj=self.c, name="x")

        for u, v in self.T.edges():
            valid_edges = [(i, j) for i in self.tree_to_graph[u] for j in self.tree_to_graph[v] if (i, j) in self.edges]
            
            self.m.addConstr(
                gp.quicksum(self.x[i, j] for i, j in valid_edges) == 1, 
                name=f"transitory_{u}_{v}"
            )

    def _callback(self, model, where):

        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(self.x)
            
            selected_edges = [e for e in self.edges if x_vals[e] > 0.5]
            solution_G = nx.DiGraph()
            solution_G.add_nodes_from(self.nodes)
            solution_G.add_edges_from(selected_edges)

            reachable_nodes = set(nx.descendants(solution_G, self.root))
            reachable_nodes.add(self.root)
            
            unreached_terminals = [t for t in self.terminals if t not in reachable_nodes]
            
            if unreached_terminals:
                for u, v in self.G.edges():
                    self.G[u][v]['capacity'] = max(0.0, x_vals.get((u, v), 0.0))
                
                added_cuts = set()
                
                for t in unreached_terminals:
                    cut_value, partition = nx.minimum_cut(self.G, self.root, t, capacity='capacity')
                    root_side, terminal_side = partition
                    
                    cut_edges = []
                    for u in root_side:
                        for v in self.G.successors(u):
                            if v in terminal_side:
                                cut_edges.append((u, v))
                                
                    if cut_edges:
                        cut_signature = frozenset(cut_edges)
                        if cut_signature not in added_cuts:
                            model.cbLazy(gp.quicksum(self.x[e] for e in cut_edges) >= 1)
                            added_cuts.add(cut_signature)

    def post_process(self, T):
        '''
        Remove any unifurcations that have 0 branch length
        '''
        unifurcations = [n for n in T if T.out_degree[n]==1 and n != self.root]

        to_remove = []
        for u in unifurcations:
            child = list(T.neighbors(u))[0]
            if self.c[u,child] == 0: 
                to_remove.append((u,child))
        
        for u,v in to_remove:
              parent = list(T.predecessors(u))
              if len(parent) > 0:
                if u.split("_")[0] == parent[0].split("_")[0]:
                    parent = parent[0]
                    T.remove_node(u)
                    T.add_edge(parent, v)

        return T
    
    def run(self):
        for t in self.terminals:
            if not nx.has_path(self.G, self.root, t):
                print(f"Model Infeasible: Terminal {t} is fundamentally unreachable in base graph G.")
                return None, None
        
        self.createModel()

        # Start the clock just before optimization begins
        self.start_time = time.time()
        self.m.optimize(self._callback)

        if self.m.Status == GRB.OPTIMAL:
            score = self.m.ObjVal
            T = nx.DiGraph()
            
            for e in self.edges:
                if self.x[e].X > 0.5:
                    T.add_edge(*e)
                    
            return score, T

        else:
            print(f"Solver stopped with status code: {self.m.Status}")
            return None, None
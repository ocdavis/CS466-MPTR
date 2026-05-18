import networkx as nx 
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import time

class MPTR_GUR:
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
        # self.timeout = timeout
        # self.start_time = None

    def createModel(self):
        self.m = gp.Model("MPTR_Original")
        self.m.Params.OutputFlag = 0
        self.m.Params.Threads = self.threads

        # --- Apply the time limit ---
        # if self.timeout is not None:
        #     self.m.Params.TimeLimit = self.timeout
        
        # 1. Variables
        # obj=self.c automatically sets the objective function to minimize edge costs
        self.x = self.m.addVars(self.edges, vtype=GRB.BINARY, obj=self.c, name="x")
        
        # Continuous flow variables f[t, u, v]
        flow_indices = [(t, u, v) for t in self.terminals for u, v in self.edges]
        self.f = self.m.addVars(flow_indices, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        # 2. Flow Upper Bound: Flow can only travel on selected edges
        self.m.addConstrs(
            (self.f[t, u, v] <= self.x[u, v] for t, u, v in flow_indices), 
            name="flow_bound"
        )

        # 3. Transitory Constraint: Pick exactly one graph edge for each tree edge
        for u, v in self.T.edges():
            valid_edges = [(i, j) for i in self.tree_to_graph[u] for j in self.tree_to_graph[v] if (i, j) in self.edges]
            self.m.addConstr(
                gp.quicksum(self.x[i, j] for i, j in valid_edges) == 1, 
                name=f"transitory_{u}_{v}"
            )

        # 4. Flow Conservation Constraints
        for t in self.terminals:
            for v in self.nodes:
                incoming_edges = [(i, v) for i in self.G.predecessors(v) if (i, v) in self.edges]
                outgoing_edges = [(v, o) for o in self.G.successors(v) if (v, o) in self.edges]
                
                in_flow = gp.quicksum(self.f[t, i, v] for i, v in incoming_edges)
                out_flow = gp.quicksum(self.f[t, v, o] for v, o in outgoing_edges)
                
                if v == self.root:
                    self.m.addConstr(out_flow == 1, name=f"flow_root_out_{t}_{v}")
                    
                elif v in self.terminals:
                    # Pyomo implementation skips constraints for terminals that are NOT the destination `t`.
                    if v == t:
                        self.m.addConstr(in_flow == 1, name=f"term_in_{t}")
                        if len(outgoing_edges) > 0:
                            self.m.addConstr(out_flow == 0, name=f"term_out_{t}")
                            
                else:
                    # Internal node flow conservation
                    self.m.addConstr(in_flow == out_flow, name=f"flow_cons_{t}_{v}")

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
        self.createModel()
        
        self.start_time = time.time()
        self.m.optimize()

        if self.m.Status == GRB.OPTIMAL:
            score = self.m.ObjVal
            T_out = nx.DiGraph()
            
            for e in self.edges:
                # In Gurobi, you access variable values via `.X`
                if self.x[e].X > 0.5:
                    T_out.add_edge(*e)
                    
            return score, T_out
            
        # elif self.m.Status in [GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        #     print(f"Solver timed out after {time.time() - self.start_time:.1f} seconds.")
        #     return None, None
            
        else:
            print(f"Solver stopped with status code: {self.m.Status}")
            return None, None
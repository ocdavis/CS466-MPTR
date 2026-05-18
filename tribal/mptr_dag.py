import pyomo
import pyomo.environ as pyo
import pyomo.opt
import networkx as nx 
import numpy as np 
import logging

class MPTR_DAG:
    def __init__(self, G, T, S, edge_weight, tree_to_graph, root=0, threads=3) -> None:
        
        self.G= G
        self.T = T
        self.terminals = S 
        self.tree_to_graph = tree_to_graph
        self.root= root 
        self.nodes = set(G.nodes)
        self.edges = set(G.edges)
        self.c = {e: edge_weight[e]for e in self.edges}
        self.internal_nodes = [n for n in self.nodes if n not in self.terminals and n!= self.root ]
        self.flow_dest = {(t,e) for e in self.edges for t in self.terminals}


    def createModel(self):
 
             
        self.m = pyo.ConcreteModel()

        self.m.x = pyo.Var(self.edges, domain=pyo.Binary)

        self.m.y = pyo.Var(self.edges, domain=pyo.Binary)

        def obj_rule(m):
            return sum(self.c[e] *m.x[e] for e in self.edges)
        
        self.m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        def transitory_rule(m, u, v):
            return sum(m.x[i, j] for i in self.tree_to_graph[u] for j in self.tree_to_graph[v] if (i, j) in self.edges) == 1

        self.m.TransitoryConstraint = pyo.Constraint(self.T.edges, rule=transitory_rule)


        def incoming_edge_consistency_rule(m, u, v):
            if u == self.root:
                return pyo.Constraint.Skip
            
            incoming = list(self.G.predecessors(u))

            return sum(m.x[i,u] for i in incoming) >= m.x[u,v]
        
        self.m.IncomingEdgeConsistency = pyo.Constraint(self.edges, rule=incoming_edge_consistency_rule)


        # def terminal_incoming_one_rule(m, u):
        #     if u == self.root:
        #         return pyo.Constraint.Skip
            
        #     incoming = list(self.G.predecessors(u))

        #     return sum(m.x[i,u] for i in incoming) == 1
        
        # self.m.TerminalIncomingOne = pyo.Constraint(self.terminals, rule=terminal_incoming_one_rule)
        

    def post_process(self, T):
    #  return T
        '''
        Remove any unifurcations that have 0 branch length
        '''
        unifurcations = [n for n in T if T.out_degree[n]==1 and n != self.root]

        to_remove = []
        for u in unifurcations:
            
            child = list(T.neighbors(u))[0]

            if self.c[u,child] ==0: # and self.seq_weights[u,child]==0:
                to_remove.append((u,child))
        
        for u,v in to_remove:
              parent = list(T.predecessors(u))
              if len(parent) > 0:
                if u.split("_")[0] ==parent[0].split("_")[0]:
                    parent = parent[0]
                    T.remove_node(u)
                    T.add_edge(parent, v)


        return T
                   
 
    
    
    def run(self):
     

            logging.getLogger('pyomo.core').setLevel(logging.ERROR)
            self.createModel()
            solver = pyomo.opt.SolverFactory("glpk")
    
            results = solver.solve(self.m, tee=False, keepfiles=False)

            if (results.solver.status != pyomo.opt.SolverStatus.ok):
                print('Check solver not ok?')
            if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
                print('Check solver optimality?') 

            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                solution = {e: pyo.value(self.m.x[e]) for e in self.edges}
                # flow = {(t, e): pyo.value(self.m.f[t, e]) for t, e in self.flow_dest}
                score = pyo.value(self.m.OBJ)  # Assuming you have an objective defined
                T = nx.DiGraph()
                for e in self.edges:         
                    if solution[e] > 0.5 :
                        T.add_edge(*e)
        
            return score,T
        
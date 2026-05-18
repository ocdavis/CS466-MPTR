import time
import pandas as pd
import numpy as np
import pickle
import os
from tribal import BaseTree
from tribal.multi_expansion import ConstructGraphMulti
from tribal.lineage_tree import MPTR as MPTR_original
from tribal.mptr_cp_gur import MPTR_CP_GUR

def run_benchmark_comparison(data_file="data/benchmark_multi_data_3.pkl", output_csv="results/benchmark_results_5_18_gur_r1.csv",resume=True):
    print(f"Loading benchmark data from {data_file}...")
    with open(data_file, "rb") as f:
        dataset = pickle.load(f)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    write_header = True

    print(f"Starting benchmark on {len(dataset)} total cases...")

    for i, case in enumerate(dataset):

        print(f"[{i+1}/{len(dataset)}] Running Case: {case.id}")
        
        base_tree = BaseTree(case.tree, root=case.root, id=0, name=case.id)
        cg = ConstructGraphMulti(case.states, case.characters, root_identifier=case.root)
        fg = cg.build(base_tree)

        # 1. Original MPTR ILP
        start_t = time.perf_counter()
        mptr_orig = MPTR_original(
            fg.G,
            base_tree.T,
            fg.find_terminals(),
            fg.iso_weights,
            fg.tree_to_graph,
            root=case.root,
        )
        score_orig, _ = mptr_orig.run()
        time_original = time.perf_counter() - start_t

        print(f"Orig time: {time_original}")

        # 2. Cutting Plane MPTR ILP
        start_t = time.perf_counter()
        mptr_cp = MPTR_CP_GUR(
            fg.G,
            base_tree.T,
            fg.find_terminals(),
            fg.iso_weights,
            fg.tree_to_graph,
            root=case.root,
        )
        score_cp, _ = mptr_cp.run()
        time_cp = time.perf_counter() - start_t

        print(f"Cutting Planes time: {time_cp}")

        current_result = {
            "id": case.id,
            "seed": case.seed,
            "n_leaves": case.n_leaves,
            "n_states": case.n_states,
            "char_1": case.char_1,
            "char_2": case.char_2,
            "preserve_triangle": case.preserve_triangle,
            "nodes_in_contracted_tree": case.tree.number_of_nodes(),
            "expansion_nodes": fg.G.number_of_nodes(),
            "expansion_edges": fg.G.number_of_edges(),
            # "time_original": time_original,
            "time_gur": time_original,
            "time_cp": time_cp,
            # "score_original": score_orig if score_orig is not None else np.nan,
            "score_gur": score_orig if score_orig is not None else np.nan,
            "score_cp": score_cp if score_cp is not None else np.nan,
            "score_match": np.isclose(score_orig, score_cp, atol=1e-4) if (score_orig is not None and score_cp is not None) else False,
            "speedup_factor": time_original / (time_cp + 1e-9)
        }
        
        df_row = pd.DataFrame([current_result])
        df_row.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False

    print(f"\nAll cases finished! Full results saved to {output_csv}")

if __name__ == "__main__":
    run_benchmark_comparison()
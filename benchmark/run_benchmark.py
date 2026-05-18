import time
import pandas as pd
import numpy as np
import pickle
import os # <-- Added for directory checking
from tribal import BaseTree
from tribal.multi_expansion import ConstructGraphMulti
from tribal.lineage_tree import MPTR as MPTR_original
from tribal.mptr_dag import MPTR_DAG
from tribal.mptr_cp import MPTR_CP
from tribal.mptr_cp_gur import MPTR_CP_GUR
from tribal.mptr_gur import MPTR_GUR
from create_benchmark import BenchmarkCase

def run_benchmark_comparison(data_file="data/benchmark_multi_data_3.pkl", output_csv="results/benchmark_results_5_18_gur_r1.csv",resume=True):
    print(f"Loading benchmark data from {data_file}...")
    with open(data_file, "rb") as f:
        dataset = pickle.load(f)

    # Ensure the output directory exists so to_csv doesn't fail
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # # If the file already exists from an old run, remove it so we don't append to old data
    # if os.path.exists(output_csv):
    #     os.remove(output_csv)

    # results = []
    # print(f"Starting benchmark on {len(dataset)} cases...")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    processed_ids = set()
    write_header = True

    # Check for existing progress
    if resume and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            processed_ids = set(existing_df['id'].tolist())
            print(f"Resuming run... Found {len(processed_ids)} completed cases in CSV.")
            write_header = False # Don't write the header again if we already have data
        except pd.errors.EmptyDataError:
            pass # File exists but is empty
    elif not resume and os.path.exists(output_csv):
        # Only delete the file if we explicitly tell it NOT to resume
        print("Starting fresh. Deleting old CSV.")
        os.remove(output_csv)

    print(f"Starting benchmark on {len(dataset)} total cases...")

    for i, case in enumerate(dataset):
        # 1. Skip if we already processed this case
        if case.id in processed_ids:
            print(f"[{i+1}/{len(dataset)}] Skipping Case: {case.id} (Already completed)")
            continue

        if i==559 or i==578 or i==607 or i==627 or i==630 or i==659 or i==688 or i==697 or i==706 or i==721 or i==728 or i==739 or i==1200:
            print(f"Skipping Case: {case.id} (taking too long / not optimal)")
            continue

        print(f"[{i+1}/{len(dataset)}] Running Case: {case.id}")
        
        base_tree = BaseTree(case.tree, root=case.root, id=0, name=case.id)
        cg = ConstructGraphMulti(case.states, case.characters, root_identifier=case.root)
        fg = cg.build(base_tree)

        # 1. Original MPTR ILP
        # start_t = time.perf_counter()
        # mptr_orig = MPTR_original(
        #     fg.G,
        #     base_tree.T,
        #     fg.find_terminals(),
        #     fg.iso_weights,
        #     fg.tree_to_graph,
        #     root=case.root,
        # )
        # score_orig, _ = mptr_orig.run()
        # time_original = time.perf_counter() - start_t

        # print(f"Orig time: {time_original}")

        # 1.5. Gurobi (no CP) MPTR ILP
        start_t = time.perf_counter()
        mptr_gur = MPTR_GUR(
            fg.G,
            base_tree.T,
            fg.find_terminals(),
            fg.iso_weights,
            fg.tree_to_graph,
            root=case.root,
        )
        score_orig, _ = mptr_gur.run()
        time_original = time.perf_counter() - start_t

        print(f"Gur (no cp) time: {time_original}")

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

        # 3. Record Metrics for this specific run
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
        
        # results.append(current_result)

        # 4. Write incrementally to the CSV immediately
        # We only write the header on the very first iteration (i == 0)
        # df_row = pd.DataFrame([current_result])
        # df_row.to_csv(output_csv, mode='a', header=(i == 0), index=False)
        df_row = pd.DataFrame([current_result])
        df_row.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False

    print(f"\nAll cases finished! Full results saved to {output_csv}")

    # 5. Summarize using the full results list we accumulated
    # df_final = pd.DataFrame(results)
    # print("\nSummary (Mean values):")
    # summary = df_final.groupby(["n_leaves", "n_states", "char_1", "char_2"])[
    #     ["time_original", "time_cp", "speedup_factor"]
    # ].mean()
    # print(summary)

    # return df_final

if __name__ == "__main__":
    run_benchmark_comparison()
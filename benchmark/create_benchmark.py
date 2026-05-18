import itertools
import pickle
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict
from simulate_tree import random_rooted_binary_tree, make_reversible_character, make_irreversible_character, make_tree_character, simulate_on_tree, contract_tree_by_characters

@dataclass
class BenchmarkCase:
    id: str
    seed: int
    n_leaves: int
    n_states: int
    char_1: str
    char_2: str
    preserve_triangle: bool
    tree: nx.DiGraph
    root: int
    states: Dict
    characters: Dict

def generate_benchmark_dataset():
    # Method Parameters
    output_file = "data/benchmark_data_1.pkl"
    leaves_list = [4,10]
    states_list = [4, 7]
    char_types = ["reversible", "irreversible", "tree"]
    char_pairs = list(itertools.combinations_with_replacement(char_types, 3))
    triangle_flags = [True, False]
    n_instances = 5
    
    
    grid = list(itertools.product(leaves_list, states_list, char_pairs, triangle_flags, triangle_flags, triangle_flags))
    total_runs = len(grid) * n_instances
    dataset = []
    
    print(f"Generating benchmark suite with {total_runs} targeted cases...")
    
    case_counter = 0
    for leaves, states, c_pair, triangle_1, triangle_2, triangle_3 in grid:
        for instance in range(n_instances):
            seed = hash(f"{leaves}_{states}_{c_pair[0]}_{c_pair[1]}_{c_pair[2]}_{triangle_1}{triangle_2}{triangle_3}_{instance}") % (2**32)
            case_id = f"L{leaves}_S{states}_{c_pair[0]}_{c_pair[1]}_{c_pair[2]}_Tri{triangle_1}{triangle_2}{triangle_3}_i{instance}"
            
            # 1. Generate Tree
            tree, root = random_rooted_binary_tree(leaves, seed=seed)

            # 2. Generate 3 Characters (First two are targets, 3rd is noise for contraction)
            types_to_make = [c_pair[0], c_pair[1], c_pair[2]]
            characters = {}

            for i, c_type in enumerate(types_to_make):
                triangle = {}
                if i==0:
                    triangle = triangle_1
                elif i==1:
                    triangle = triangle_2
                else:
                    triangle = triangle_3

                char_name = f"char_{i}"
                if c_type == "reversible":
                    characters[char_name] = make_reversible_character(char_name, states, preserve_triangle=triangle)
                elif c_type == "irreversible":
                    characters[char_name] = make_irreversible_character(char_name, states, preserve_triangle=triangle)
                elif c_type == "tree":
                    characters[char_name] = make_tree_character(char_name, states, preserve_triangle=triangle)

            # 3. Simulate States
            rng = np.random.default_rng(seed)
            tree, sim_states = simulate_on_tree(tree, root, characters, rng=rng)

            # 4. Contract Tree (Keep char_0 and char_1, collapse on char_2)
            chars_to_contract = ["char_0", "char_1"]
            c_tree, c_states, c_chars = contract_tree_by_characters(
                tree, sim_states, characters, chars_to_contract
            )

            # Skip if contraction destroyed the tree
            if c_tree.number_of_nodes() < 2:
                continue

            # 5. Store Case
            case = BenchmarkCase(
                id=case_id,
                seed=seed,
                n_leaves=leaves,
                n_states=states,
                char_1=c_pair[0],
                char_2=c_pair[1],
                preserve_triangle=triangle,
                tree=c_tree,
                root=root,
                states=c_states,
                characters=c_chars
            )
            dataset.append(case)
            
            case_counter += 1
            if case_counter % 20 == 0:
                print(f"Generated {case_counter} valid cases...")

    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Saved {len(dataset)} valid benchmark cases to {output_file}")

if __name__ == "__main__":
    generate_benchmark_dataset()
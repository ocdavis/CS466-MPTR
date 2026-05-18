# CS466 Multi-Character Phylogenetic Inference

The method name MPTR stands for Multi-Character Polytomy Tree Refinement.

To use the simulations, install the conda environment via the following:

```bash
conda env create -f mptr.yml 
```

An example of the entire process can be seen in the python notebook `sim_tree.ipynb` for simulating trees with different character types, and running both the multi-commodity flow and cutting-plane ILP formulations for the multi-character polytomy refinement problem.

For benchmarking please refer to the files (`create_benchmark.py` and `run_benchmark.py`). Adjust the parameters (listed under the "Method Parameters" comment) before running.

Create benchmarks with

```bash
python create_benchmark.py
```

Run a benchmark with

```bash
python run_benchmark.py
```


## Reference

This project uses some base code from TRIBAL, see:

Weber, L. L., Reiman, D., Roddur, M. S., Qi, Y., El-Kebir, M., & Khan, A. A. TRIBAL: Tree Inference of B cell Clonal Lineages. bioRxiv. [doi.org/10.1101/2023.11.27.568874](https://doi.org/10.1101/2023.11.27.568874).




## License

[BSD-3](license.md)

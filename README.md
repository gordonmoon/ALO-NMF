# ALO-NMF

The C++ and CUDA implementations of **ALO-NMF** described in the paper titled, "ALO-NMF: Accelerated Locality-Optimized Non-negative Matrix Factorization".

## Dependencies
- Intel Compiler (The C++ code is optimized on Intel CPUs)
- CUDA Compiler (The CUDA code is optimized on NVIDIA Tesla P100 PCIE GPU)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
  
## Prepare datasets
Download datasets: `./data.sh`

## Compile the codes
To compile the program, type the following command in each directory: `make clean` and `make`

## Runtime usage
```
export MKL_NUM_THREADS=48
export OMP_NUM_THREADS=48
```

## BASH script options
- {K}: Low rank
- {tile_size}: Tile size 'T'
- {data}: Non-negative input matrix 'A'
- {matrix_type}: type of 'A'. 1 - Dense matrix, 2 - Sparse matrix
- {V}: Number of rows in 'A'
- {D}: Number of columns in 'A'
- {niters}: Number of iterations

## Run the codes
To run the program, type the following command in each directory:
  + Run ALO-NMF CPU: `./run_alo_nmf_cpu.sh`
  + Run ALO-NMF GPU: `./run_alo_nmf_gpu.sh`
  
## Citation
If you use ALO-NMF in a scientific publication, we would appreciate citations to the following paper:
```
@inproceedings{moon2020alo,
  title={ALO-NMF: Accelerated Locality-Optimized Non-negative Matrix Factorization},
  author={Moon, Gordon E and Ellis, J Austin and Sukumaran-Rajam, Aravind and Parthasarathy, Srinivasan and Sadayappan, P},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1758--1767},
  year={2020}
}
```

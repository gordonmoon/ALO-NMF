# ALO-NMF

The C++ and CUDA implementations of **ALO-NMF** described in the paper titled, "ALO-NMF: Accelerated Locality-Optimized Non-negative Matrix Factorization".

## Dependencies
- Intel Compiler (The C++ code is optimized on Intel CPUs)
- CUDA Compiler (The CUDA code is optimized on NVIDIA Tesla P100 PCIE GPU)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
  
## Data
Download sample sparse and dense matrices: `./data.sh`

## Compile the codes
Type the following commands in `ALO-NMF_CPU` and `ALO-NMF_GPU` directories.
```
make clean
make 
```

## Runtime usage
```
export MKL_NUM_THREADS=48
export OMP_NUM_THREADS=48
```

## BASH script options
Specify the options in `./run_alo_nmf_cpu.sh` and `./run_alo_nmf_gpu.sh`
- `K`: Low rank
- `tile_size`: Tile size 'T', given a 'K' value, the tile size 'T' needs to be one of the factors of K (e.g., when K = 64, possible T values are 1, 2, 4, 8, 16, 32 and 64). The code will be updated to use an arbitrary tile size.
- `data`: Non-negative input matrix 'A'
- `matrix_type`: type of 'A'. 1 - Dense matrix, 2 - Sparse matrix
- `V`: Number of rows in 'A'
- `D`: Number of columns in 'A'
- `niters`: Number of iterations

## Run the codes
  + Run ALO-NMF CPU:
  ```
  cd ALO-NMF_CPU
  ./run_alo_nmf_cpu.sh
  ```
  + Run ALO-NMF GPU:
  ```
  cd ALO-NMF_GPU
  ./run_alo_nmf_gpu.sh
  ```
  
## Citation
If you use this ALO-NMF in your own work, please cite our paper as follows:
```
@inproceedings{moon2020alo,
  title={ALO-NMF: Accelerated Locality-Optimized Non-negative Matrix Factorization},
  author={Moon, Gordon E and Ellis, J Austin and Sukumaran-Rajam, Aravind and Parthasarathy, Srinivasan and Sadayappan, P},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1758--1767},
  year={2020}
}
```

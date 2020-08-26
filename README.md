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

## Run the codes
To run the program, type the following command in each directory:
  + Run ALO-NMF CPU: `./run_nmf_hals_cpu.sh`
  + Run ALO-NMF GPU: `./run_nmf_hals_gpu.sh`
  
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

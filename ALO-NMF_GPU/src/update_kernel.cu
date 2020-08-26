#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "update_kernel.h"
#include "cuda_util.h"
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

using namespace CUDAUtil;


__global__ void update_phase_two_H(double const* _d_H_old, double* _d_H_new, double const* _d_temp_s, double const* _d_temp_r, int t, int tile_id, int Tile_size, int D, int K, double eps) {

    int dId = blockIdx.x*blockDim.x + threadIdx.x; // each thread will take dot product of each row of H and column of temp_s

    if (dId >= D) {
        return;
    }
    double tmp = 0;
    for (int k = tile_id*Tile_size ; k < (tile_id+1)*Tile_size; k++) {
        if(k < t) {
            tmp += _d_H_new[dId+k*D] * _d_temp_s[k*K+t];
        }
        else {
            tmp += _d_H_old[dId+k*D] * _d_temp_s[k*K+t];
        }
    }

    _d_H_new[dId+t*D] = max(_d_H_new[dId+t*D] - tmp + _d_temp_r[dId+t*D],eps);

}


__global__ void update_m_W_q(double const* _d_W_old, double* _d_W_new, double const* _d_temp_q, int V, int K) {

    int vId = blockIdx.x*blockDim.x + threadIdx.x;

    if (vId >= V) {
        return;
    }

    for (int k = 0; k < K; k++) {
        _d_W_new[vId+k*V] = _d_W_old[vId+k*V]*_d_temp_q[k+k*K];
    }

}


__global__ void update_phase_two_W(double const* _d_W_old, double* _d_W_new, double const* _d_temp_q, double const* _d_temp_p, double* _d_ss_col, int t, int tile_id, int Tile_size, int V, int K, double eps) {

    int vId = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ double s_r[1024/32];
    double r = 0.0f;

    if (vId < V) {

        double tmp = 0;
        for (int k = tile_id*Tile_size ; k < (tile_id+1)*Tile_size; k++) {
            if(k < t) {
                tmp += _d_W_new[vId+k*V] * _d_temp_q[k*K+t];
            }
            else {
                tmp += _d_W_old[vId+k*V] * _d_temp_q[k*K+t];
            }
        }
        r = _d_W_new[vId+t*V] = max(_d_W_new[vId+t*V] - tmp + _d_temp_p[vId+t*V],eps);

    }

    r = r*r;
    // warp-level reduction
    r += __shfl_down(r, 16);
    r += __shfl_down(r, 8);
    r += __shfl_down(r, 4);
    r += __shfl_down(r, 2);
    r += __shfl_down(r, 1);
    // block-level reduction
    if(threadIdx.x % 32 == 0)
        s_r[threadIdx.x/32] = r;
    __syncthreads();
    if(threadIdx.x / 32 == 0) {
        r = s_r[threadIdx.x];
    }
    __syncthreads();
    r += __shfl_down(r, 16);
    r += __shfl_down(r, 8);
    r += __shfl_down(r, 4);
    r += __shfl_down(r, 2);
    r += __shfl_down(r, 1);

    if(threadIdx.x == 0)
        atomicAdd(&_d_ss_col[0], r);

}


__global__ void update_d_W_col(double* _d_W_new, double* _d_ss_col, int t, int V) {

    int vId = blockIdx.x*blockDim.x + threadIdx.x;

    if (vId >= V) {
        return;
    }

    _d_W_new[vId+t*V] = _d_W_new[vId+t*V]/sqrt(_d_ss_col[0]);
}


void cuda_mul_W_old_temp_q(double* _d_W_old, double* _d_W_new, double* _d_temp_q, int V, int K) {

    int numBlock = (V+1023)/1024;
    int numThread = 1024;
    update_m_W_q<<<numBlock,numThread>>>(_d_W_old, _d_W_new, _d_temp_q, V, K);

}

void phase_two_H(double* _d_H_old, double* _d_H_new, double* _d_temp_s, double* _d_temp_r, int t, int tile_id, int Tile_size, int D, int K, double eps) {

    int numBlock = (D+1023)/1024;
    int numThread = 1024;
    update_phase_two_H<<<numBlock,numThread>>>(_d_H_old, _d_H_new, _d_temp_s, _d_temp_r, t, tile_id, Tile_size, D, K, eps);

}

void phase_two_W(double* _d_W_old, double* _d_W_new, double* _d_temp_q, double* _d_temp_p, double* _d_ss_col, int t, int tile_id, int Tile_size, int V, int K, double eps) {

    int numBlock = (V+1023)/1024;
    int numThread = 1024;
    update_phase_two_W<<<numBlock,numThread>>>(_d_W_old, _d_W_new, _d_temp_q, _d_temp_p, _d_ss_col, t, tile_id, Tile_size, V, K, eps);
    
}

void cuda_div_W_new_col(double* _d_W_new, double* _d_ss_col, int t, int V) {

    int numBlock = (V+1023)/1024;
    int numThread = 1024;
    update_d_W_col<<<numBlock,numThread>>>(_d_W_new, _d_ss_col, t, V);
    
}
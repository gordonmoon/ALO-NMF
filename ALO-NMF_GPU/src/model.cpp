#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cmath>
#include <cassert>
#include <typeinfo>
#include <map>
#include "constants.h"
#include "utils.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector> 
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <memory.h>
#include <string.h>
#include "cuda_util.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "update_kernel.h"

#define EPSILON_1EMINUS16 0.00000000000000001

using namespace std;
using namespace CUDAUtil;


inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    if(result != cudaSuccess)
        exit (-1);
  }
  return result;
}

double rtclock(void) {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

bool myfunction (int i,int j) { return (i<j); }

struct myclass {
    bool operator() (int i,int j) { return (i<j);}
} myobject;

model::~model() {
    if (model_status == MODEL_STATUS_ALO_NMF_GPU) {

    }
}

void model::set_default_values() {
    model_status = MODEL_STATUS_UNKNOWN;
    V = 0;
    D = 0;
    K = 0;
    niters = 0;
    liter = 0;
    matrix_type = 0;
}

int model::parse_args(int argc, char ** argv) {
    return utils::parse_args(argc, argv, this);
}

int model::init(int argc, char ** argv) {
    if (parse_args(argc, argv)) {
        return 1;
    }
    return 0;
}

int model::init_est() {
    printf("NMF initialization\n");
    return 0;
}

template <class T>
void fixNumericalError(T *X, const double prec = EPSILON_1EMINUS16) {
  (*X).for_each(
      [&](typename T::elem_type &val) { val = (val < prec) ? prec : val; });
}

void model::estimate_ALO_NMF_GPU() {

    if (matrix_type == 1) {
        printf("Input matrix is a dense matrix\n");
    }
    else if (matrix_type == 2) {
        printf("Input matrix is a sparse matrix\n");
    }

    Tile_size = TS;

    m_W_old = new double[V*K];
    m_W_new = new double[V*K];
    m_H_old = new double[D*K];
    m_H_new = new double[D*K];


    //                                              //
    //           Random initialization              //
    //                                              //

    double eps_W_H = 1e-5;
    srand48(0L);
    for (int v = 0; v < V; v++) {
        for (int k = 0; k < K; k++) {
            m_W_old[v+k*V] = 0.1 * drand48();
            if (m_W_old[v+k*V] >= 1) {
                m_W_old[v+k*V] = m_W_old[v+k*V] - eps_W_H;
            }
            if (m_W_old[v+k*V] <= 0 || m_W_old[v+k*V] >= 1) {
                printf("random intialization error \n");
            }
            m_W_new[v+k*V] = m_W_old[v+k*V];
        }
    }

    srand48(0L);
    for (int d = 0; d < D; d++) {
        for (int k = 0; k < K; k++) {
            m_H_old[d+k*D] = 0.1 * drand48();
            if (m_H_old[d+k*D] >= 1) {
                m_H_old[d+k*D] = m_H_old[d+k*D] - eps_W_H;
            }
            if (m_H_old[d+k*D] <= 0 || m_H_old[d+k*D] >= 1) {
                printf("random intialization error \n");
            }
            m_H_new[d+k*D] = m_H_old[d+k*D];
        }
    }


    //                                  //
    //       Load input matrix          //
    //                                  //


    vector <vector <double> > data;
    ifstream infile(train_file);

    while (infile)
    {
        string s;
        if (!getline( infile, s )) break;
        istringstream ss( s );
        vector <double> record;
        while (ss)
        {
            string s;
            if (!getline( ss, s, ',' )) break;
            double a = atof(s.c_str());
            record.push_back( a );
        }
        data.push_back( record );
    }

    unsigned int total_nnz = 0;
    double ss = 0.0;
    m_denseData = new double[V*D];
    for (int v = 0; v < V; v++) {
      for (int d = 0; d < D; d++) {
        m_denseData[v+d*V] = data[v][d];
        ss += m_denseData[v+d*V]*m_denseData[v+d*V];
        if (m_denseData[v+d*V] != 0) {
            total_nnz += 1;
        }
      }
    }
    data.clear();

    norm_trainData = sqrt(ss);
    printf("total number of nnz = %lld\n",total_nnz);

    if (matrix_type == 2) {

        csr_val_trainData = new double[total_nnz];
        csr_col_ind_trainData = new int[total_nnz];
        csr_row_ptr_trainData = new int[V+1];

        int nnz_idx = 0;
        for (int v = 0; v < V; v++) {
            csr_row_ptr_trainData[v] = nnz_idx;
            for (int d = 0; d < D; d++) {
                if (m_denseData[v+d*V] != 0) {
                    csr_val_trainData[nnz_idx] = m_denseData[v+d*V];
                    csr_col_ind_trainData[nnz_idx] = d;
                    nnz_idx++;
                }
            }
        }
        csr_row_ptr_trainData[V] = nnz_idx;

        csr_val_trainData_T = new double[total_nnz];
        csr_col_ind_trainData_T = new int[total_nnz];
        csr_row_ptr_trainData_T = new int[D+1];

        int nnz_idx_T = 0;
        for (int d = 0; d < D; d++) {
            csr_row_ptr_trainData_T[d] = nnz_idx_T;
            for (int v = 0; v < V; v++) {
                if (m_denseData[v+d*V] != 0) {
                    csr_val_trainData_T[nnz_idx_T] = m_denseData[v+d*V];
                    csr_col_ind_trainData_T[nnz_idx_T] = v;
                    nnz_idx_T++;
                }
            }
        }
        csr_row_ptr_trainData_T[D] = nnz_idx_T;
    }

    if (matrix_type == 1) {
        cudaMalloc((void**)&_d_denseData, sizeof (double) *(V*D));
        cudaMemcpy(_d_denseData, m_denseData, sizeof (double) *(V*D), cudaMemcpyHostToDevice);
    }
    else if (matrix_type == 2) {
        cudaMalloc((void**)&_d_csr_val, sizeof (double) *total_nnz);
        cudaMemcpy(_d_csr_val, csr_val_trainData, sizeof (double) *total_nnz, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&_d_csr_col_ind, sizeof (int) *total_nnz);
        cudaMemcpy(_d_csr_col_ind, csr_col_ind_trainData, sizeof (int) *total_nnz, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&_d_row_ptr, sizeof (int) *(V+1));
        cudaMemcpy(_d_row_ptr, csr_row_ptr_trainData, sizeof (int) *(V+1), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&_d_csr_val_T, sizeof (double) *total_nnz);
        cudaMemcpy(_d_csr_val_T, csr_val_trainData_T, sizeof (double) *total_nnz, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&_d_csr_col_ind_T, sizeof (int) *total_nnz);
        cudaMemcpy(_d_csr_col_ind_T, csr_col_ind_trainData_T, sizeof (int) *total_nnz, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&_d_row_ptr_T, sizeof (int) *(D+1));
        cudaMemcpy(_d_row_ptr_T, csr_row_ptr_trainData_T, sizeof (int) *(D+1), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&_d_H_old, sizeof (double) *(D*K));
    cudaMalloc((void**)&_d_H_new, sizeof (double) *(D*K));
    cudaMemcpy(_d_H_old, m_H_old, sizeof (double) *(D*K), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_H_new, m_H_new, sizeof (double) *(D*K), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_temp_r, sizeof (double) *(D*K));
    cudaMemset((void *)_d_temp_r,0, sizeof (double) *(D*K));
    cudaMalloc((void**)&_d_temp_s, sizeof (double) *(K*K));
    cudaMemset((void *)_d_temp_s,0, sizeof (double) *(K*K));

    cudaMalloc((void**)&_d_W_old, sizeof (double) *(V*K));
    cudaMalloc((void**)&_d_W_new, sizeof (double) *(V*K));
    cudaMemcpy(_d_W_old, m_W_old, sizeof (double) *(V*K), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_W_new, m_W_new, sizeof (double) *(V*K), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_d_temp_p, sizeof (double) *(V*K));
    cudaMemset((void *)_d_temp_p,0, sizeof (double) *(V*K));
    cudaMalloc((void**)&_d_temp_q, sizeof (double) *(K*K));
    cudaMemset((void *)_d_temp_q,0, sizeof (double) *(K*K));

    cudaMalloc((void**)&_d_ss_col, sizeof (double) *1);
    cudaMemset((void *)_d_ss_col,0, sizeof (double) *1);

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    int num_tiles = (K+Tile_size-1) / Tile_size;
    printf("Number of tile = %d\n",num_tiles);
    printf("Tile size = %d\n",Tile_size);

    double rel_errors;
    rel_errors = compute_rel_error();
    printf("initial relative error = %f\n",rel_errors);

    int n_epoch = niters;
    int print_error_step = 1;
    int iter_id = 0;
    double* elapsed_time_iter;
    elapsed_time_iter = new double[n_epoch/print_error_step];
    for (int et = 0; et < n_epoch/print_error_step; et++) {
        elapsed_time_iter[et] = 0.0;
    }
    double* error_iter;
    error_iter = new double[n_epoch/print_error_step];
    for (int ep = 0; ep < n_epoch/print_error_step; ep++) {
        error_iter[ep] = 0.0;
    }

    double sum_total_time = 0.0;
    double eps = 1e-16;
    char transa;
    char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};

    printf("%d iterations (GPU, parallel)\n", n_epoch);

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        printf("Iteration %d ...\n", epoch+1);
        rel_errors = 0.0;
        float mili =0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMemcpy(_d_W_old, _d_W_new, sizeof (double) *(V*K), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_d_H_old, _d_H_new, sizeof (double) *(D*K), cudaMemcpyDeviceToDevice);

/********************************updating H************************************/

        alpha_cuda = 1.0; beta_cuda = 0.0;

        if (matrix_type == 1) {
            cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, K, V, &alpha_cuda, _d_denseData, V, _d_W_old, V, &beta_cuda, _d_temp_r, D); // for dense dataset
            cudaDeviceSynchronize();
        }
        else if (matrix_type == 2) {
            cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, D, K, V, total_nnz, &alpha_cuda, descr, _d_csr_val_T, _d_row_ptr_T, _d_csr_col_ind_T, _d_W_old, V, &beta_cuda, _d_temp_r, D); // for sparse dataset
            cudaDeviceSynchronize();
        }

        cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, K, V, &alpha_cuda, _d_W_old, V, _d_W_old, V, &beta_cuda, _d_temp_s, K);
        cudaDeviceSynchronize();


        // update H - PHASE 1
        alpha_cuda = -1.0; beta_cuda = 1.0;

        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, (tile_id*Tile_size), Tile_size, &alpha_cuda, _d_H_old+(tile_id*Tile_size*D), D, _d_temp_s+(tile_id*Tile_size*K), K, &beta_cuda, _d_H_new, D);
            cudaDeviceSynchronize();
        }

        // update H - PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                phase_two_H(_d_H_old, _d_H_new, _d_temp_s, _d_temp_r, t, tile_id, Tile_size, D, K, eps);
                cudaDeviceSynchronize();
            }
            if (tile_id < num_tiles-1){
                cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, (K-(tile_id+1)*Tile_size), Tile_size, &alpha_cuda, _d_H_new+(tile_id*Tile_size*D), D, _d_temp_s+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, &beta_cuda, _d_H_new+((tile_id+1)*Tile_size*D), D);
                cudaDeviceSynchronize();
            }
        }

/********************************updating W************************************/

        alpha_cuda = 1.0; beta_cuda = 0.0;

        if (matrix_type == 1) {
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, V, K, D, &alpha_cuda, _d_denseData, V, _d_H_new, D, &beta_cuda, _d_temp_p, V); // for dense dataset
            cudaDeviceSynchronize();
        }
        else if (matrix_type == 2) {
            cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, V, K, D, total_nnz, &alpha_cuda, descr, _d_csr_val, _d_row_ptr, _d_csr_col_ind, _d_H_new, D, &beta_cuda, _d_temp_p, V); // for sparse dataset
            cudaDeviceSynchronize();
        }

        cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, K, D, &alpha_cuda, _d_H_new, D, _d_H_new, D, &beta_cuda, _d_temp_q, K);
        cudaDeviceSynchronize();

        cuda_mul_W_old_temp_q(_d_W_old, _d_W_new, _d_temp_q, V, K);
        cudaDeviceSynchronize();

        // update W - PHASE 1
        alpha_cuda = -1.0; beta_cuda = 1.0;
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, (tile_id*Tile_size), Tile_size, &alpha_cuda, _d_W_old+(tile_id*Tile_size*V), V, _d_temp_q+(tile_id*Tile_size*K), K, &beta_cuda, _d_W_new, V);
            cudaDeviceSynchronize();
        }

        // update W - PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {

                cudaMemset((void *)_d_ss_col,0, sizeof (double) *1);
                phase_two_W(_d_W_old, _d_W_new, _d_temp_q, _d_temp_p, _d_ss_col, t, tile_id, Tile_size, V, K, eps);
                cudaDeviceSynchronize();

                cuda_div_W_new_col(_d_W_new, _d_ss_col, t, V);
                cudaDeviceSynchronize();

            }
            if (tile_id < num_tiles-1){
                cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, (K-(tile_id+1)*Tile_size), Tile_size, &alpha_cuda, _d_W_new+(tile_id*Tile_size*V), V, _d_temp_q+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, &beta_cuda, _d_W_new+((tile_id+1)*Tile_size*V), V);
                cudaDeviceSynchronize();
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&mili, start, stop);
        sum_total_time += mili;


        if ((epoch+1) % print_error_step == 0) {
            cudaMemcpy(m_H_new, _d_H_new, sizeof (double) *(D*K), cudaMemcpyDeviceToHost);
            cudaMemcpy(m_W_new, _d_W_new, sizeof (double) *(V*K), cudaMemcpyDeviceToHost);
            rel_errors = compute_rel_error();
            printf("relative error = %f\n",rel_errors);
            error_iter[iter_id] = rel_errors;

            printf("elpased time = %f\n", sum_total_time/1000);
            elapsed_time_iter[iter_id] = sum_total_time/1000;
            iter_id++;
        }

    }
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    cublasDestroy(handle_cublas);


    free(m_denseData);

    free(m_H_old);
    free(m_H_new);
    free(m_W_old);
    free(m_W_new);

    if (matrix_type == 1) {
        cudaFree(_d_denseData);
    }
    else if (matrix_type == 2) {
        free(csr_val_trainData);
        free(csr_col_ind_trainData);
        free(csr_row_ptr_trainData);
        free(csr_val_trainData_T);
        free(csr_col_ind_trainData_T);
        free(csr_row_ptr_trainData_T);
        cudaFree(_d_csr_val);
        cudaFree(_d_csr_col_ind);
        cudaFree(_d_row_ptr);
        cudaFree(_d_csr_val_T);
        cudaFree(_d_csr_col_ind_T);
        cudaFree(_d_row_ptr_T);
    }

    cudaFree(_d_H_old);
    cudaFree(_d_H_new);
    cudaFree(_d_temp_r);
    cudaFree(_d_temp_s);
    cudaFree(_d_W_old);
    cudaFree(_d_W_new);
    cudaFree(_d_temp_p);
    cudaFree(_d_temp_q);
    cudaFree(_d_ss_col);

    // printf("\n===========ELASPED TIME===========\n\n");
    ofstream myfile_time;
    myfile_time.open ("ALO-NMF_GPU_time.txt");
    for (int output_time =0; output_time < n_epoch/print_error_step; output_time++) {
        myfile_time << elapsed_time_iter[output_time] << endl;
        // printf("%f\n",elapsed_time_iter[output_time]);
    }
    myfile_time.close();

    // printf("\n===========RELATIVE ERROR===========\n\n");
    ofstream myfile_error;
    myfile_error.open ("ALO-NMF_GPU_rel_error.txt");
    for (int output_error = 0; output_error < n_epoch/print_error_step; output_error++) {
        myfile_error << error_iter[output_error] << endl;
        // printf("%f\n",error_iter[output_error]);
    }
    myfile_error.close();

    printf("Accumulated Total Time: %fs\n", sum_total_time/1000);

}

double model::compute_rel_error() {

    double rel_error = 0.0;
    double norm_error = 0.0;
    double ts = 0.0;

    #pragma omp parallel for reduction(+:ts)
    for (int v = 0; v < V; v++) {
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += m_W_new[k*V+v]*m_H_new[k*D+d];
            }
            ts += (m_denseData[v+d*V]-sum)*(m_denseData[v+d*V]-sum);
        }
    }

    norm_error = sqrt(ts);
    rel_error = norm_error / norm_trainData;

    return rel_error;
}
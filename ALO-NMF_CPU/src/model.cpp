#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl.h"
#include "mkl_spblas.h"
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

#define EPSILON_1EMINUS16 0.00000000000000001

using namespace std;

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
    if (model_status == MODEL_STATUS_ALO_NMF_CPU) {

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

void model::estimate_ALO_NMF_CPU() {

    if (matrix_type == 1) {
        printf("Input matrix is a dense matrix\n");
    }
    else if (matrix_type == 2) {
        printf("Input matrix is a sparse matrix\n");
    }

    Tile_size = TS;

    m_W_old = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_W_new = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_H_old = (double *)mkl_malloc( D*K*sizeof( double ), 64 );
    m_H_new = (double *)mkl_malloc( D*K*sizeof( double ), 64 );

    m_temp_p = (double *)mkl_malloc( V*K*sizeof( double ), 64 );
    m_temp_q = (double *)mkl_malloc( K*K*sizeof( double ), 64 );
    m_temp_r = (double *)mkl_malloc( D*K*sizeof( double ), 64 );
    m_temp_s = (double *)mkl_malloc( K*K*sizeof( double ), 64 );

    //                                              //
    //           Random initialization              //
    //                                              //

    double eps_W_H = 1e-5;
    srand48(0L);
    for (int v = 0; v < V; v++) {
        for (int k = 0; k < K; k++) {
            m_W_old[v*K+k] = 0.1 * drand48();
            if (m_W_old[v*K+k] >= 1) {
                m_W_old[v*K+k] = m_W_old[v*K+k] - eps_W_H;
            }
            m_W_new[v*K+k] = m_W_old[v*K+k];
        }
    }

    srand48(0L);
    for (int d = 0; d < D; d++) {
        for (int k = 0; k < K; k++) {
            m_H_old[d*K+k] = 0.1 * drand48();
            if (m_H_old[d*K+k] >= 1) {
                m_H_old[d*K+k] = m_H_old[d*K+k] - eps_W_H;
            }
            m_H_new[d*K+k] = m_H_old[d*K+k];
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
    m_denseData = (double *)mkl_malloc( V*D*sizeof( double ), 64 );

    double ss = 0.0;
    for (int v = 0; v < V; v++) {
      for (int d = 0; d < D; d++) {
        m_denseData[v*D+d] = data[v][d];
        ss += m_denseData[v*D+d]*m_denseData[v*D+d];
        if (m_denseData[v*D+d] != 0) {
            total_nnz += 1;
        }
      }
    }

    norm_trainData = sqrt(ss);
    data.clear();

    printf("total number of nnz = %lld\n",total_nnz);

    if (matrix_type == 2) {
        csr_val_trainData = new double[total_nnz];
        csr_col_trainData = new int[total_nnz];
        csr_ptrB_trainData = new int[V];
        csr_ptrE_trainData = new int[V];

        int nnz_idx = 0;
        for (int v = 0; v < V; v++) {
            csr_ptrB_trainData[v] = nnz_idx;
            for (int d = 0; d < D; d++) {
                if (m_denseData[v*D+d] != 0) {
                    csr_val_trainData[nnz_idx] = m_denseData[v*D+d];
                    csr_col_trainData[nnz_idx] = d;
                    nnz_idx++;
                }
            }
            csr_ptrE_trainData[v] = nnz_idx;
        }

        trainData_sparse = (double *)mkl_malloc( total_nnz*sizeof(double), 64);
        trainData_sparse_cols = (MKL_INT *)mkl_malloc(total_nnz*sizeof(MKL_INT), 64);
        trainData_sparse_ptrB = (MKL_INT *)mkl_malloc(V*sizeof(MKL_INT),64);
        trainData_sparse_ptrE = (MKL_INT *)mkl_malloc(V*sizeof(MKL_INT),64);

        for (int i = 0; i < total_nnz; i++) {
            trainData_sparse[i] = csr_val_trainData[i];
            trainData_sparse_cols[i] = csr_col_trainData[i];
        }

        for (int v = 0; v < V; v++) {
            trainData_sparse_ptrB[v] = csr_ptrB_trainData[v];
            trainData_sparse_ptrE[v] = csr_ptrE_trainData[v];
        }
    }

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

    printf("%d iterations (CPU, parallel)\n", n_epoch);

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        printf("Iteration %d ...\n", epoch+1);
        rel_errors = 0.0;
        double start = rtclock();

        // Copy old matrices from previous iteration

        #pragma omp parallel for
        for (int v = 0; v< V; v++) {
            for (int k = 0; k < K; k++) {
                m_W_old[v*K+k] = m_W_new[v*K+k];
            }
        }
        #pragma omp parallel for
        for (int d = 0; d< D; d++) {
            for (int k = 0; k < K; k++) {
                m_H_old[d*K+k] = m_H_new[d*K+k];
            }
        }

/********************************updating H************************************/

        transa = 'T';
        MKL_INT ldb = K, ldc=K;
        alpha_mkl = 1.0; beta_mkl = 0.0;

        if (matrix_type == 1) {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, K, V, alpha_mkl, &m_denseData[0], D, &m_W_old[0], K, beta_mkl, &m_temp_r[0], K); // for dense dataset
        }
        else if (matrix_type == 2) {
            mkl_dcsrmm(&transa, &V, &K, &D, &alpha_mkl, matdescra, trainData_sparse, trainData_sparse_cols, trainData_sparse_ptrB, trainData_sparse_ptrE, &m_W_old[0], &ldb, &beta_mkl, &m_temp_r[0], &ldc); // for sparse dataset
        }

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, V, alpha_mkl, &m_W_old[0], K, &m_W_old[0], K, beta_mkl, &m_temp_s[0], K);

        // PHASE 1
        alpha_mkl = -1.0; beta_mkl = 1.0;

        for (int tile_id = 1; tile_id < num_tiles; tile_id++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, (tile_id*Tile_size), Tile_size, alpha_mkl, &m_H_old[0]+(tile_id*Tile_size), K, &m_temp_s[0]+(tile_id*Tile_size*K), K, beta_mkl, &m_H_new[0], K);
        }

        // PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                #pragma omp parallel for
                for (int d = 0; d < D; d++) {
                    double tmp = 0;
                    int k = tile_id*Tile_size;
                    #pragma omp simd reduction(+:tmp)
                    for (; k < t; k++) {
                            tmp += (m_H_new[d*K+k]*m_temp_s[t*K+k]);
                    }
                    #pragma omp simd reduction(+:tmp)
                    for (k=t; k < (tile_id+1)*Tile_size; k++) {
                            tmp += (m_H_old[d*K+k]*m_temp_s[t*K+k]);
                    }
                    m_H_new[d*K+t] = max(m_H_new[d*K+t] - tmp + m_temp_r[d*K+t],eps);
                }
            }
            if (tile_id < num_tiles-1){
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, (K-(tile_id+1)*Tile_size), Tile_size, alpha_mkl, &m_H_new[0]+(tile_id*Tile_size), K, &m_temp_s[0]+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, beta_mkl, &m_H_new[0]+((tile_id+1)*Tile_size), K);
            }
        }


/********************************updating W************************************/

        alpha_mkl = 1.0; beta_mkl = 0.0;
        transa = 'N';

        if (matrix_type == 1) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, K, D, alpha_mkl, &m_denseData[0], D, &m_H_new[0], K, beta_mkl, &m_temp_p[0], K); // for dense dataset
        }
        else if (matrix_type == 2) {
            mkl_dcsrmm(&transa, &V, &K, &D, &alpha_mkl, matdescra, trainData_sparse, trainData_sparse_cols, trainData_sparse_ptrB, trainData_sparse_ptrE, &m_H_new[0], &ldb, &beta_mkl, &m_temp_p[0], &ldc); // for sparse dataset
        }

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, D, alpha_mkl, &m_H_new[0], K, &m_H_new[0], K, beta_mkl, &m_temp_q[0], K);

        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            for (int k = 0; k < K; k++) {
                m_W_new[v*K+k] = m_W_old[v*K+k]*m_temp_q[k*K+k];
            }
        }

        // PHASE 1
        alpha_mkl = -1.0; beta_mkl = 1.0;

        for (int tile_id = 1; tile_id < num_tiles; tile_id++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, (tile_id*Tile_size), Tile_size, alpha_mkl, &m_W_old[0]+(tile_id*Tile_size), K, &m_temp_q[0]+(tile_id*Tile_size*K), K, beta_mkl, &m_W_new[0], K);
        }

        // PHASE 2 & 3
        for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
            for (int t = tile_id*Tile_size; t < (tile_id+1)*Tile_size; t++) {
                double ss_col = 0;

                #pragma omp parallel for reduction(+:ss_col)
                for (int v = 0; v < V; v++) {
                    double tmp = 0;
                    int k = tile_id*Tile_size;
                    #pragma omp simd reduction(+:tmp)
                    for (; k < t; k++) {
                            tmp += (m_W_new[v*K+k]*m_temp_q[t*K+k]);
                    }
                    #pragma omp simd reduction(+:tmp)
                    for (k=t; k < (tile_id+1)*Tile_size; k++) {
                            tmp += (m_W_old[v*K+k]*m_temp_q[t*K+k]);
                    }
                    m_W_new[v*K+t] = max(m_W_new[v*K+t] - tmp + m_temp_p[v*K+t],eps);
                    ss_col += m_W_new[v*K+t]*m_W_new[v*K+t];
                }

                #pragma omp parallel for
                for (int w = 0; w < V; w++) {
                    m_W_new[w*K+t] = m_W_new[w*K+t]/sqrt(ss_col);
                }
            }
            if (tile_id < num_tiles-1){
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, V, (K-(tile_id+1)*Tile_size), Tile_size, alpha_mkl, &m_W_new[0]+(tile_id*Tile_size), K, &m_temp_q[0]+((tile_id*Tile_size*K)+((tile_id+1)*Tile_size)), K, beta_mkl, &m_W_new[0]+((tile_id+1)*Tile_size), K);
            }
        }

        double end = rtclock();
        sum_total_time += end-start;

        if ((epoch+1) % print_error_step == 0) {

            rel_errors = compute_rel_error();
            printf("relative error = %f\n",rel_errors);
            error_iter[iter_id] = rel_errors;

            printf("elpased time = %f\n", sum_total_time);
            elapsed_time_iter[iter_id] = sum_total_time;
            iter_id++;

        }
    }

    mkl_free(m_W_old);
    mkl_free(m_W_new);
    mkl_free(m_H_old);
    mkl_free(m_H_new);

    mkl_free(m_temp_p);
    mkl_free(m_temp_q);
    mkl_free(m_temp_r);
    mkl_free(m_temp_s);

    mkl_free(m_denseData);

    if (matrix_type == 2) {
        free(csr_val_trainData);
        free(csr_col_trainData);
        free(csr_ptrB_trainData);
        free(csr_ptrE_trainData);

        mkl_free(trainData_sparse);
        mkl_free(trainData_sparse_cols);
        mkl_free(trainData_sparse_ptrB);
        mkl_free(trainData_sparse_ptrE);
    }

    // printf("\n===========ELASPED TIME===========\n\n");
    ofstream myfile_time;
    myfile_time.open ("ALO-NMF_CPU_time.txt");
    for (int output_time =0; output_time < n_epoch/print_error_step; output_time++) {
        myfile_time << elapsed_time_iter[output_time] << endl;
        // printf("%f\n",elapsed_time_iter[output_time]);
    }
    myfile_time.close();

    // printf("\n===========RELATIVE ERROR===========\n\n");
    ofstream myfile_error;
    myfile_error.open ("ALO-NMF_CPU_relative_error.txt");
    for (int output_error = 0; output_error < n_epoch/print_error_step; output_error++) {
        myfile_error << error_iter[output_error] << endl;
        // printf("%f\n",error_iter[output_error]);
    }
    myfile_error.close();

    printf("Accumulated Total Time: %fs\n\n", sum_total_time);
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
                sum += m_W_new[v*K+k]*m_H_new[d*K+k];
            }
            ts += (m_denseData[v*D+d]-sum)*(m_denseData[v*D+d]-sum);
        }
    }

    norm_error = sqrt(ts);
    rel_error = norm_error / norm_trainData;

    return rel_error;
}
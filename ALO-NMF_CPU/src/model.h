#ifndef _MODEL_H
#define _MODEL_H

#include "mkl.h"
#include "constants.h"
#include <iostream>

using namespace std;

// NMF model
class model {
public:
    // fixed options
    int model_status;
    // --- model parameters and variables ---
    string model_name;

    double alpha_mkl, beta_mkl;
    int matrix_type;
    int V;
    int D;
    int K;
    int Tile_size;

    double norm_trainData;

    int TS;
    int niters;
    int liter;
    string train_file;
    string * vocabmap;

    // For CPU implementation
    double *m_W_old, *m_W_new, *m_H_old, *m_H_new, *m_temp_p, *m_temp_q, *m_temp_r, *m_temp_s;
    double *m_denseData;
    double *trainData_sparse;
    MKL_INT *trainData_sparse_cols;
    MKL_INT *trainData_sparse_ptrB;
    MKL_INT *trainData_sparse_ptrE;
    
    double *csr_val_trainData;
    int *csr_col_trainData;
    int *csr_ptrB_trainData;
    int *csr_ptrE_trainData;

    model() {
    set_default_values();
    }

    ~model();

    void set_default_values();
    int parse_args(int argc, char ** argv);
    int init(int argc, char ** argv);
    int init_est();

    void estimate_ALO_NMF_CPU();
    double compute_rel_error();

};

#endif

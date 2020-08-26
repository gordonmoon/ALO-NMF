#ifndef _MODEL_H
#define _MODEL_H

#include "constants.h"
#include <iostream>

using namespace std;

// NMF model
class model {
public:
    // fixed options

    int model_status;       // model status:
    // --- model parameters and variables ---

    string model_name;

    double alpha_cuda, beta_cuda;
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

    // For GPU implementation

    double *m_W_old, *m_W_new, *m_H_old, *m_H_new;
    double *m_denseData;

    double* csr_val_trainData;
    int* csr_col_ind_trainData;
    int* csr_row_ptr_trainData;

    double* csr_val_trainData_T;
    int* csr_col_ind_trainData_T;
    int* csr_row_ptr_trainData_T;

    double *_d_csr_val, *_d_csr_val_T;
    int *_d_csr_col_ind, *_d_row_ptr, *_d_csr_col_ind_T, *_d_row_ptr_T;

    double *_d_denseData, *_d_H_old, *_d_H_new, *_d_temp_r, *_d_temp_s;
    double *_d_W_old, *_d_W_new, *_d_temp_p, *_d_temp_q, *_d_ss_col;

    model() {
    set_default_values();
    }

    ~model();

    void set_default_values();
    int parse_args(int argc, char ** argv);
    int init(int argc, char ** argv);
    int init_est();

    void estimate_ALO_NMF_GPU();
    double compute_rel_error();

};

#endif

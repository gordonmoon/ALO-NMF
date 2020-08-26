#include <stdio.h>
#include <string>
#include <map>
#include "utils.h"
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

int utils::parse_args(int argc, char ** argv, model * pmodel) {
    int model_status = MODEL_STATUS_UNKNOWN;
    int K = 0;
    int TS = 0;
    int niters = 0;
    string train_file = "";
    int V = 0;
    int D = 0;
    int matrix_type = 0;

    int i = 0;
    while (i < argc) {
    string arg = argv[i];

    if (arg == "-est_nmf_cpu") {
        model_status = MODEL_STATUS_ALO_NMF_CPU;

    } else if (arg == "-K") {
        K = atoi(argv[++i]);

    } else if (arg == "-tile_size") {
        TS = atoi(argv[++i]);

    } else if (arg == "-data") {
        train_file = argv[++i];

    } else if (arg == "-matrix_type") {
        matrix_type = atoi(argv[++i]);

    } else if (arg == "-V") {
        V = atoi(argv[++i]);

    } else if (arg == "-D") {
        D = atoi(argv[++i]);

    } else if (arg == "-niters") {
        niters = atoi(argv[++i]);

    } else {
        // any more?
    }

    i++;
    }

    if (model_status == MODEL_STATUS_ALO_NMF_CPU) {
        pmodel->model_status = model_status;
        if (K > 0) {
            pmodel->K = K;
        }
        if (TS > 0) {
            pmodel->TS = TS;
        }
        if (niters > 0) {
            pmodel->niters = niters;
        }
        if (matrix_type > 0) {
            pmodel->matrix_type = matrix_type;
        }
        if (V > 0) {
            pmodel->V = V;
        }
        if (D > 0) {
            pmodel->D = D;
        }
        pmodel->train_file = train_file;
    }

    if (model_status == MODEL_STATUS_UNKNOWN) {
        printf("Please specify the task you would like to perform (-est/-estc/-inf)!\n");
        return 1;
    }

    return 0;
}

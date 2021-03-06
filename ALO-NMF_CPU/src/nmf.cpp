#include "model.h"

#include <cstdio>
#include <cstdlib>

using namespace std;

void show_help();

int main(int argc, char ** argv) {

    srandom(0); // initialize for random number generation

    model nmf;

    if (nmf.init(argc, argv)) {
        show_help();
        return 1;
    }

    if (nmf.model_status == MODEL_STATUS_ALO_NMF_CPU) {
        printf("ALO-NMF on multi-core CPUs\n");
        nmf.estimate_ALO_NMF_CPU();
    }

    return 0;
}

void show_help() {
    printf("Command line usage:\n");
}


#ifndef _UTILS_H
#define _UTILS_H

#include <string>
#include <vector>

using namespace std;

class model;

class utils {
public:
    // parse command line arguments
    static int parse_args(int argc, char ** argv, model * pmodel);

};

#endif

#ifndef ALO_NMF_CUDA_UTIL_H
#define	ALO_NMF_CUDA_UTIL_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

/*constants*/
#define NUM_TOTAL_THREAD	(gridDim.x*blockDim.x)
#define GLOBAL_THREAD_OFFSET	(blockDim.x*blockIdx.x + threadIdx.x)
#define WARP_SIZE		(32)	//be carefule when using this

namespace CUDAUtil {

    /*timing functions*/
    //return ms, rather than second!

    inline void startTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    inline float endTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return elapsedTime;
    }

    class Timer {
    protected:
        float _t; //ms
    public:

        Timer() :
        _t(0.0f) {
        };

        virtual ~Timer() {
        };

        virtual void go() = 0;
        virtual void stop() = 0;

        void reset() {
            _t = 0;
        }

        float report() const {
            return _t;
        }
    };

    class CPUTimer : public Timer {
    private:
        struct timeval _start, _end;
    public:

        CPUTimer() :
        _start(), _end() {
        }

        ~CPUTimer() {
        }

        void go() {
            gettimeofday(&_start, NULL);
        }

        void stop() {
            gettimeofday(&_end, NULL);
            _t += ((_end.tv_sec - _start.tv_sec)*1000.0f + (_end.tv_usec - _start.tv_usec)/1000.0f);
        }
    };

    class CUDATimer : public Timer {
    private:
        cudaEvent_t _start, _stop;
    public:

        CUDATimer() : _start(), _stop(){
        };

        inline void go() {
            startTimer(_start, _stop);
        };

        inline void stop() {
            _t += endTimer(_start, _stop);
        };
    };

    /*CUDA helper functions*/

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
        if (cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int) err, cudaGetErrorString(err));
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int) err, cudaGetErrorString(err));
            exit(-1);
        }
    }


} /*namespace CUDAUtil*/

#endif	/* MIAN_CUDA_UTIL_H */
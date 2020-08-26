# GPU architecture specification
GPU_ARCH_FLAG	= arch=compute_60,code=sm_60

# C++ compiler configuration
CXX             = g++
CXXFLAGS        = -O3 -fopenmp -DBH=128 -DBW=64

# CUDA compiler configuration
NVCC_HOME      	= /home/xxxx/cuda/9.2.88
NVCC            = nvcc
CUDA_INC        = -I$(NVCC_HOME)/include
CUDA_LIB        = -L$(NVCC_HOME)/lib64 -lcudart -lcublas -lcusparse
CUDA_FLAGS		= -O3 -m64 -gencode $(GPU_ARCH_FLAG) -std=c++11 -DBH=128 -DBW=64

INCLUDE     	= $(CUDA_INC)
LIB     		= $(CUDA_LIB)
OBJS			= utils.o model.o nmf.o update_kernel.o
MAIN			= nmf

all: $(OBJS)
	$(CXX) -o $(MAIN) $(OBJS) ${LIB} ${CXXFLAGS}
utils.o: ./src/utils.h ./src/utils.cpp
	$(CXX) -c -o utils.o ./src/utils.cpp $(CXXFLAGS) $(INCLUDE)
model.o: ./src/model.h ./src/model.cpp
	$(CXX) -c -o model.o ./src/model.cpp $(CXXFLAGS) $(INCLUDE)
update_kernel.o: ./src/update_kernel.h ./src/update_kernel.cu
	$(NVCC) -c ./src/update_kernel.cu $(INCLUDE) $(CUDA_FLAGS)
nmf.o: ./src/nmf.cpp
	$(CXX) -c -o nmf.o ./src/nmf.cpp $(CXXFLAGS) $(INCLUDE)

clean:
	rm $(OBJS)
	rm $(MAIN)
# C++ compiler configuration
CXX             = icc
CXXFLAGS        = -O3 -fopenmp -mkl -DBH=128 -DBW=64
OBJS			= utils.o model.o nmf.o
MAIN			= nmf

all: $(OBJS)
	$(CXX) -o $(MAIN) $(OBJS) ${LIB} ${CXXFLAGS}
utils.o: ./src/utils.h ./src/utils.cpp
	$(CXX) -c -o utils.o ./src/utils.cpp $(CXXFLAGS)
model.o: ./src/model.h ./src/model.cpp
	$(CXX) -c -o model.o ./src/model.cpp $(CXXFLAGS)
nmf.o: ./src/nmf.cpp
	$(CXX) -c -o nmf.o ./src/nmf.cpp $(CXXFLAGS)

clean:
	rm $(OBJS)
	rm $(MAIN)
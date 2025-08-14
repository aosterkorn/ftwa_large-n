CPP_BASE = $(wildcard base/*.cpp)
CPP_SU_N = $(wildcard su_n/*.cpp)

OBJECTS = $(CPP_BASE:.cpp=.o) $(CPP_SU_N:.cpp=.o)

TARGETS_SU_N = $(basename $(wildcard su_*.cpp))
# TARGETS_SU_N = su_n_hubhei_quench su_n_hubhei_peierls su_n_test

CXXFLAGS = -Wall -std=c++14 -O2 -DARMA_ALLOW_FAKE_GCC -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG # -DFTWA_WITH_TIMER# -DFTWA_CACHE_CHECKPOINTS 
# -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -DFTWA_CACHE_CHECKPOINTS
LDFLAGS = -lopenblas -llapack -lhdf5 -lboost_program_options

 
CXX = g++

all: $(TARGETS_SU_N)

su_%: su_%.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OBJECTS) $(LDFLAGS)

# .PRECIOUS: %.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS_SU_N) $(OBJECTS)

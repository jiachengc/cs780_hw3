CC = g++
CUDA = nvcc
CFLAGS = -Wall -Wextra -std=c++11 -lpthread -pedantic -O3 -ldl -pthread # for linux
# CFLAGS = -Wall -Wextra -std=c++11 -lpthread -pedantic -O3 -pthread # for windows MinGW
CUDAFLAGS = -std=c++11 -arch=sm_75

dot_product.o: dot_product.cu dot_product.h
	nvcc -O2 -std=c++14 --use_fast_math --compiler-options '-fPIC' --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__ -D_GLIBCXX_USE_CXX11_ABI=0  -c -o dot_product.o dot_product.cu 

net_ext.o: net_ext.cpp dot_product.h
	g++ -O2 -fPIC -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -c -o net_ext.o net_ext.cpp  $(shell python3-config --includes)

net_ext.so: net_ext.o dot_product.o
	nvcc -shared -o net_ext.so net_ext.o dot_product.o  $(shell python3-config --libs)
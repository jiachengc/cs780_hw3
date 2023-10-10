# cs780_hw3

Python Extension Library Description
Contains two source files: dot_product.cu and net_ext.cpp.
dot_product.cu mainly implements CUDA multiplication operations, and net_ext.cpp mainly encapsulates the Python interface.

Compiling the Python Extension Library
Run: make net_ext.so
This will generate the net_ext.so file in the current directory.

Simple Test for the Python Extension Library
Run: python3 net_ext_test.py

To compile, you need Python header files. You can get the path to the Python header files using the following command:
python3-config --includes

Running the mnist_dnn for Training and Testing
Run: python3 main.py epoch_numbers

Tip: It will increase the Testing acc as epoch goes larger. for me 300 epoch could get 85.2% acc.

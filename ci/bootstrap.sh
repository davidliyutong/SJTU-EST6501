#!/bin/bash

ROOT=$(pwd)

echo "[INFO] python training"
cd MNIST_PYTHORCH_C/tests/train_model
echo "[INFO] cd MNIST_PYTHORCH_C"
python main.py --epochs=1
cd $ROOT
echo "[INFO] cd ROOT"

echo "[INFO] cmake build"
if [ ! -d "MNIST_PYTHORCH_C/build" ]; then
echo "[INFO] making dir MNIST_PYTHORCH_C/build" && mkdir -p "MNIST_PYTHORCH_C/build"
fi
cd MNIST_PYTHORCH_C/build
cmake ..
make
echo "[INFO] Runing mnist_example"
time ./mnist_example > mnist_example.log
echo "[INFO] Runing test_compiler"
time ./test_compiler > test_compiler.log
cd $ROOT

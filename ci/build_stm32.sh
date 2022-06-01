#!/bin/bash

ROOT=$(pwd)

echo "[INFO] Copy model parmas, functions and test data"
cp -rf MNIST_PYTHORCH_C/tests/train_model/export_code_stm32 MNIST_PYTHORCH_C/tests/test_stm32/export_code

echo "[INFO] cd MNIST_PYTHORCH_C/tests/test_stm32"
cd MNIST_PYTHORCH_C/tests/test_stm32
make

cd $ROOT
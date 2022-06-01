#!/bin/bash

ROOT=$(pwd)

if [ ! -d "MNIST_PYTHORCH_C/tests/test_stm32/export_code/" ]; then
echo "[INFO] making dir MNIST_PYTHORCH_C/tests/test_stm32/export_code/" && mkdir -p "MNIST_PYTHORCH_C/tests/test_stm32/export_code/"
fi

echo "[INFO] Copy model parmas, functions and test data"
cp -rf MNIST_PYTHORCH_C/tests/train_model/export_code_stm32/* MNIST_PYTHORCH_C/tests/test_stm32/export_code/

echo "[INFO] cd MNIST_PYTHORCH_C/tests/test_stm32"
cd MNIST_PYTHORCH_C/tests/test_stm32
make

cd $ROOT
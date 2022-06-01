#!/bin/bash

ROOT=$(pwd)

echo "[INFO] python training"
cd MNIST_PYTHORCH_C/tests/train_model
echo "[INFO] cd MNIST_PYTHORCH_C"
python main_stm32.py --epochs=6
cd $ROOT
echo "[INFO] cd ROOT"

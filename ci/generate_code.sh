#!/bin/bash

ROOT=$(pwd)

echo "[INFO] python training"
cd MNIST_PYTHORCH_C/tests/train_model
echo "[INFO] cd MNIST_PYTHORCH_C"
python main.py --epochs=1
cd $ROOT
echo "[INFO] cd ROOT"
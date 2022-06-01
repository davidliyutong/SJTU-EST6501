#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
cd $DIR/../
ROOT=$(pwd)

rm -rf ./build
rm -rf ./MNIST_PYTHORCH_C/build
rm -rf ./MNIST_PYTHORCH_C/tests/test_stm32/export_code
rm -rf ./MNIST_PYTHORCH_C/tests/test_stm32/build
rm -rf ./MNIST_PYTHORCH_C/tests/train_model/export_code
rm -rf ./MNIST_PYTHORCH_C/tests/train_model/export_code_stm32
rm -rf ./MNIST_PYTHORCH_C/tests/train_model/export_model
rm -rf ./MNIST_PYTHORCH_C/tests/train_model/export_model_stm32
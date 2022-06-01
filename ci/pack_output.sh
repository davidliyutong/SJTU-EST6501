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

tar -zvcf mnist_pythorch.tar.gz ./MNIST_PYTHORCH_C/build/test_compiler \
                                ./MNIST_PYTHORCH_C/build/mnist_example \
                                ./MNIST_PYTHORCH_C/tests/test_stm32/build/stm32l475vgtx.elf \
                                ./MNIST_PYTHORCH_C/tests/train_model/export_code \
                                ./MNIST_PYTHORCH_C/tests/train_model/export_code_stm32 \
                                ./MNIST_PYTHORCH_C/tests/train_model/export_model \
                                ./MNIST_PYTHORCH_C/tests/train_model/export_model_stm32
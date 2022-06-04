#!/bin/bash
URL=https://github-davidliyutong.oss-cn-hangzhou.aliyuncs.com/SJTU-EST6501/data.tar.gz
curl $URL -o data.tar.gz
tar -xzf data.tar.gz -C ./MNIST_PYTHORCH_C/tests/train_model/
rm data.tar.gz
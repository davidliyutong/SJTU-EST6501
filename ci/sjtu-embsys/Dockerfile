from ubuntu:focal

WORKDIR /opt/embsys
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip git curl cmake clang wget gcc ninja-build&& \
    apt-get clean

RUN curl https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2 -o gcc-arm-none-eabi.tar.bz2 && \
    tar -xjf gcc-arm-none-eabi.tar.bz2 && \
    rm gcc-arm-none-eabi.tar.bz2 && \
    mv gcc-arm-none-eabi-10.3-2021.10 /usr/local/bin/gcc-arm-none-eabi

RUN pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple && \
    pip install torch==1.10 matplotlib ipython torch-summary iotlabcli

RUN ln -s $(which python3) /usr/local/bin/python

ENV PATH /usr/local/bin/gcc-arm-none-eabi/bin:$PATH

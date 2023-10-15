# hash:sha256:67f7e721ee601310c2da37edf879d814e8829d849cee9ed39158a35af7bc0091
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.7.2 \
        numpy==1.26.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    cvxpy==1.4.0 \
    mosek==10.1.13

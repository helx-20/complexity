# hash:sha256:7bbb1a0150f2ddcce5d1eeb7ceb31c25a77dcd3b3bb6e4cecdfaa6d6b0de2f84
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    gym==0.25.0 \
    matplotlib==3.7.3 \
    mujoco==2.2.0 \
    numpy==1.24.4 \
    pandas==1.4.1 \
    pillow==8.4.0 \
    scipy==1.10.1 \
    tensorboard==2.14.0 \
    thop==0.1.1.post2209072238 \
    torch==1.10.2 \
    torch-optimizer==0.3.0 \
    torchaudio==0.10.2 \
    torchstat==0.0.7 \
    torchvision==0.11.3

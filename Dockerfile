# Navier-Stokes 2D AI-HPC Hybrid Solver
# Docker image for reproducible builds

FROM ubuntu:22.04

LABEL maintainer="AI-HPC Project"
LABEL description="Navier-Stokes 2D Hybrid Solver with OpenMP and PyTorch"

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgomp1 \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Build C++ solver with OpenMP
RUN mkdir -p build && \
    g++ -std=c++17 -O3 -fopenmp -DUSE_OPENMP \
        -I include src/core/*.cpp src/main.cpp \
        -o build/ns_main_omp -lpthread && \
    g++ -std=c++11 -O2 \
        src/win_parallel_benchmark.cpp \
        -o build/parallel_benchmark

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    matplotlib \
    pandas \
    scipy

# Create results directory
RUN mkdir -p results checkpoints data/checkpoints

# Set environment variables
ENV OMP_NUM_THREADS=4
ENV PYTHONPATH=/app/python

# Default command
CMD ["/bin/bash"]

# Example usage:
# docker build -t ns-solver .
# docker run -it --rm ns-solver
# docker run -it --rm ns-solver ./build/ns_main_omp 1.0 128 0.01
# docker run -it --rm ns-solver python3 python/visualize.py

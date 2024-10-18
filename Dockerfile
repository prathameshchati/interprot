FROM python:3.12

WORKDIR /app
COPY . /app

# Install dependencies for mmseqs2
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    zlib1g-dev \
    bzip2 \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install mmseqs2
RUN wget https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz \
    && tar xvzf mmseqs-linux-avx2.tar.gz \
    && mv mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs mmseqs-linux-avx2.tar.gz

# Verify installation
RUN mmseqs version

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

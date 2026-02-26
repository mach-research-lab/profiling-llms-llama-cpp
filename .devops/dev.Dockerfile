ARG PYTHON_VERSION=3.14.3

# ---------------------------------------------------------------------------
# Build stage: compile llama.cpp (CPU-only) on Ubuntu 24.04 with PAPI headers
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS build

ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        libpapi-dev \
        papi-tools && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_TESTS=OFF && \
    cmake --build build -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full && \
    cp build/bin/* /app/full && \
    cp *.py /app/full && \
    cp -r gguf-py /app/full && \
    cp -r requirements /app/full && \
    cp requirements.txt /app/full && \
    cp .devops/tools.sh /app/full/tools.sh

# ---------------------------------------------------------------------------
# Runtime stage: Ubuntu 24.04 with PAPI runtime, Python 3.14.3 from source
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG PYTHON_VERSION

RUN apt-get update && \
    apt-get install -y \
        libgomp1 \
        curl \
        libpapi7.1t64 \
        papi-tools \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        libffi-dev \
        liblzma-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* /var/tmp/* \
        /var/cache/apt/archives /var/lib/apt/lists

WORKDIR /tmp

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd /tmp && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

COPY --from=build /app/lib/ /app/
COPY --from=build /app/full/ /app/

WORKDIR /app

# torch~=2.6.0 has no Python 3.14 wheel; relax the upper bound at build time
RUN sed -i 's/torch~=2\.6\.0/torch>=2.9.0/' requirements/requirements-convert_hf_to_gguf.txt && \
    python3.14 -m pip install --upgrade pip setuptools wheel && \
    python3.14 -m pip install -r requirements.txt

ENTRYPOINT ["/app/tools.sh"]

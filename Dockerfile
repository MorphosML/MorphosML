# -------- Builder --------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    make \
    pybind11-dev\
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install scikit-build-core pybind11 numpy pytest

# Copy only necessary files first (cache optimization)
COPY pyproject.toml .
COPY . .

# Build and install
RUN pip install -v .

# -------- Runtime --------
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Test
RUN python3 -c "import morphosml; print('MorphosML loaded')"

CMD ["python3"]
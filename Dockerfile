# -------- Builder --------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gcc \
    make \
    pybind11-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir scikit-build-core pybind11 numpy pytest

# Copy repository files
COPY . .

# Build and install
RUN pip install --no-cache-dir -v .

# -------- Runtime --------
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system morphosml \
    && useradd --system --gid morphosml --create-home morphosml

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN chown morphosml:morphosml /app
USER morphosml

# Test
RUN python3 -c "import morphosml; print('MorphosML loaded')"

CMD ["python3"]
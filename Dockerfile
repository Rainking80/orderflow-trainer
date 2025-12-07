# UPDATED: Switched to Ubuntu 20.04 to support the TensorFlow wheel's glibc requirement
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# 2. Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python 3.8 and essential build tools
# Ubuntu 20.04 has Python 3.8 as default, so we simplify this step
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    build-essential \
    cmake \
    libomp-dev \
    python3-distutils \
    python3-venv \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip (using the legacy get-pip.py for Python 3.8)
# UPDATED: Changed URL to the specific version for 3.8 support
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# 5. Ensure 'python' command points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 6. Set the working directory
WORKDIR /app

# 7. Copy your requirements file first (for caching)
COPY requirements.txt .

# 8. Install your specific legacy stack
# Using the same wheel, which should now work with Ubuntu 20.04's glibc
RUN pip install https://github.com/marcossilva/tensorflow-cuda11-wheel/releases/download/v2.4.0/tensorflow-2.4.0-cp38-cp38-linux_x86_64.whl
RUN pip install -r requirements.txt

# 9. Copy the rest of your application code
COPY . .

# 11. Default command to run your script
CMD ["python", "main.py"]
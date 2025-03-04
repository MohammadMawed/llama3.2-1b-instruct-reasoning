# Use NVIDIA's CUDA base image that supports PyTorch
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app


COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the fully fine-tuned model and scripts
COPY full_model/ /app/full_model/
COPY run_model.py /app/

# Set default command to run the inference script
CMD ["python3", "run_model.py"]

# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (Git is required by Hugging Face libraries)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Keep the container running interactively by default
CMD ["/bin/bash"]
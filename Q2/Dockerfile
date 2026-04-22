FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r requirements.txt

# Copy model code and pre-trained artefacts
COPY model.py app.py ./
# Question2/ must exist (run train.py locally first)
COPY Question2/ ./Question2/

EXPOSE 7860
CMD ["python", "app.py"]

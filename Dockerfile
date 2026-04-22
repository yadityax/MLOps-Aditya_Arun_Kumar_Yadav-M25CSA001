
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translate.py .

# input.txt and reference.txt should be present in /app (copied or volume-mounted)
CMD ["python", "translate.py", "--input", "input.txt", "--reference", "reference.txt"]

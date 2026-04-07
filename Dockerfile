FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the baseline script as the default container command
# Change the last line to this:
CMD ["python", "inference.py"]
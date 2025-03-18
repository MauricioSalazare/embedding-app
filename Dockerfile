FROM python:3.11-slim

WORKDIR /app
COPY app.py requirements.txt ./
COPY data /app/data
EXPOSE 8050
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
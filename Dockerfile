# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

COPY app.py requirements.txt ./
COPY data/processed/latent_vectors_anonymized.csv /app/data/processed/latent_vectors_anonymized.csv
COPY data/processed/rlps_2023_data_anonymized.csv /app/data/processed/rlps_2023_data_anonymized.csv

EXPOSE 8050

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the model training script first, then start the Dash application
#CMD python build_sphere.py && python run.py
CMD ["sh", "-c", "python app.py"]
# Backend-Container: FastAPI + PyTorch (CPU) + Grad-CAM.
# Baut auf dem Repo-Root auf (dort liegen main.py, model.py, data.py, best_model.pth).
FROM python:3.10-slim

# System-Bibliotheken, die OpenCV (Abhängigkeit von grad-cam) braucht
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch als CPU-Wheels installieren (viel kleiner als die GPU-Variante)
RUN pip install --no-cache-dir torch==2.3.* torchvision==0.18.* \
    --index-url https://download.pytorch.org/whl/cpu

# Übrige Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode + Modellgewichte + Beispielbilder
COPY model.py data.py main.py ./
COPY best_model.pth ./
COPY samples ./samples

# In Produktion die kleinen Beispielbilder statt des großen Datensatzes nutzen
ENV TEST_DIR=samples
ENV THRESHOLD=0.5
ENV RATE_LIMIT=10
ENV RATE_WINDOW=3600

EXPOSE 8000
# EIN Worker-Prozess -> zusammen mit dem internen Worker-Thread bleibt es bei
# einer Inferenz gleichzeitig. Mehr Worker würden das umgehen und mehr RAM kosten.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Backend-Container: FastAPI + PyTorch (CPU) + Grad-CAM.
# Baut auf dem Repo-Root auf (Kontext), der Code liegt in serving/.
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
COPY serving/model/ ./model/
COPY serving/data.py serving/main.py serving/stages.py ./
# Das Klassifikatorgewicht MUSS zu MODEL_FAMILY weiter unten passen: rsna_* hat
# EINEN Logit, best_model* hat zwei Klassen. main.py prueft das beim Start gegen
# die Form von fc.weight und bricht bei Widerspruch ab, statt mit der falschen
# Vorverarbeitung zu rechnen. Wer hier die Datei tauscht, muss MODEL_FAMILY
# mittauschen.
COPY checkpoints/rsna_f4_s0.pth ./checkpoints/
COPY serving/samples ./samples

# Segmenter für die Masken-/Crop-Kachel der Pipeline-Ansicht.
# Nur unet.py wird gebraucht (die Architektur), nicht das ganze Trainingspaket.
# Fehlen diese beiden Zeilen, läuft die App weiter - die zwei Kacheln melden
# dann "übersprungen" statt ein Bild zu zeigen (siehe serving/stages.py).
COPY segmentation/__init__.py segmentation/unet.py ./segmentation/
COPY checkpoints/unet_best.pth ./checkpoints/

# Strecke und Gewicht. Beides zusammen, weil beides zusammen stimmen muss.
ENV MODEL_FAMILY=rsna
ENV CHECKPOINT_PATH=checkpoints/rsna_f4_s0.pth

# In Produktion die kleinen Beispielbilder statt des großen Datensatzes nutzen
ENV TEST_DIR=samples
ENV THRESHOLD=0.5
ENV RATE_LIMIT=10
ENV RATE_WINDOW=3600
# Vorverarbeitungskette in der Oberfläche zeigen (0 = nur Ergebnis, spart RAM/CPU)
ENV SHOW_STAGES=1

EXPOSE 8000
# EIN Worker-Prozess -> zusammen mit dem internen Worker-Thread bleibt es bei
# einer Inferenz gleichzeitig. Mehr Worker würden das umgehen und mehr RAM kosten.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

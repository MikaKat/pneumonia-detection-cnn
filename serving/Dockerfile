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
# serving/model/ enthaelt model.py UND kalibrierung_p10.json. Die JSON-Datei ist
# nicht Beiwerk: sie nennt die fuenf Gewichte, die fuenf Platt-Kurven und die
# Schwelle. Ohne sie startet main.py gar nicht erst.
COPY serving/model/ ./model/
COPY serving/data.py serving/main.py serving/stages.py ./
# ALLE FUENF Gewichte des Phase-10-Ensembles, rund 224 MB. Vier davon waeren ein
# anderes Modell als das auf dem Holdout gemessene, deshalb bricht main.py beim
# Start ab, wenn eines fehlt.
#
# Damit sie ueberhaupt im Build-Kontext liegen, braucht .gitignore fuer jede
# einzelne eine Ausnahme. Genau daran ist der Build am 26.07. schon einmal
# gescheitert: eine Zeile `*.pth` und ein COPY, das darauf zeigt.
COPY checkpoints/rsna_f0_s0_p5head_ex.pth \
     checkpoints/rsna_f1_s0_p5head_ex.pth \
     checkpoints/rsna_f2_s0_p5head_ex.pth \
     checkpoints/rsna_f3_s0_p5head_ex.pth \
     checkpoints/rsna_f4_s0_p5head_ex.pth \
     ./checkpoints/
COPY serving/samples ./samples

# Segmenter für die Lungenfinder-Karte.
# Nur unet.py wird gebraucht (die Architektur), nicht das ganze Trainingspaket.
# Fehlen diese beiden Zeilen, läuft die App weiter - die Karte entfällt dann
# einfach (siehe serving/stages.py). Der Ordner liegt seit dem 09.08.2026 unter
# serving/, weil die App sein einziger verbliebener Nutzer ist.
COPY serving/segmentation/__init__.py serving/segmentation/unet.py ./segmentation/
COPY checkpoints/unet_best.pth ./checkpoints/

# KEIN ENV CHECKPOINT_PATH und KEIN ENV MODEL_FAMILY mehr. Welche Gewichte das
# Ensemble bilden, steht in serving/model/kalibrierung_p10.json, zusammen mit
# den Kurven und der Schwelle, die zu genau diesen Gewichten gehoeren. Drei
# Angaben, die nur gemeinsam stimmen, gehoeren in eine Datei und nicht in drei
# Umgebungsvariablen.

# In Produktion die kleinen Beispielbilder statt des großen Datensatzes nutzen
ENV TEST_DIR=samples
# KEIN ENV THRESHOLD hier. Bis zum 09.08.2026 stand an dieser Stelle 0.5 und
# hat die 0.3115 aus main.py still ueberschrieben: im Protokoll stand eine
# Schwelle, im Container lief eine andere, und nichts in der Ausgabe hat
# widersprochen. Die Zeile wurde entfernt, die gleichlautende in
# docker-compose.yml aber nicht, und die haette denselben Fehler weitergefuehrt.
# Seit dem Umbau bricht main.py bei gesetztem THRESHOLD ab, statt ihn zu
# befolgen: ein Fehler, der nur durch Disziplin nicht wiederkommt, kommt wieder.
ENV RATE_LIMIT=10
ENV RATE_WINDOW=3600
# Vorverarbeitungskette in der Oberfläche zeigen (0 = nur Ergebnis, spart RAM/CPU)
ENV SHOW_STAGES=1

EXPOSE 8000
# EIN Worker-Prozess -> zusammen mit dem internen Worker-Thread bleibt es bei
# einer Inferenz gleichzeitig. Mehr Worker würden das umgehen und mehr RAM kosten.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

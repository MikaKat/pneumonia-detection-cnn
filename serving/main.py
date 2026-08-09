"""FastAPI-Backend für die Pneumonie-Web-App.

Setzt den Vertrag aus webapp/API_CONTRACT.md um:
  GET  /api/health
  GET  /api/limits
  GET  /api/samples          (+ /api/samples/{id}/thumb)
  POST /api/analyze          (Datei-Upload ODER {"sample_id": ...})
  GET  /api/jobs/{job_id}
  GET  /api/pipeline         (Beschreibung der Vorverarbeitungskette, ohne Bilder)

Kernideen:
  * EIN einziger Worker-Thread verarbeitet immer nur EIN Bild gleichzeitig.
    Alle Anfragen landen in einer Warteschlange (queue.Queue) und werden der
    Reihe nach abgearbeitet. So kann kein Ansturm mehrerer Nutzer den Server
    überlasten - selbst bei 1 GB RAM läuft nie mehr als eine Inferenz parallel.
  * Rate-Limiting im Arbeitsspeicher (pro X-Client-Id, festes Zeitfenster).
  * Das Modell wird EINMAL beim Start geladen, nicht pro Anfrage.
  * SEIT PHASE 10 IST DAS MODELL EIN ENSEMBLE. Geladen werden die fuenf
    Fold-Gewichte `rsna_f{0..4}_s0_p5head_ex.pth`, jedes wird einzeln mit
    SEINER Platt-Kurve kalibriert, und erst die fuenf WAHRSCHEINLICHKEITEN
    werden gemittelt. Kurven, Gewichtsliste und Schwelle stehen zusammen in
    `serving/model/kalibrierung_p10.json`; sie stammen ausschliesslich aus den
    Entwicklungsdaten und standen fest, bevor der Holdout gerechnet wurde.
    Genau diese Rechnung hat `rsna/pipeline/rsna_holdout.py` auf 3812
    ungesehenen Bildern ausgewertet, und genau sie wird hier ausgeliefert.
  * STUFENWEISE AUSGABE: Die Vorverarbeitung wird nicht als Block gerechnet und
    am Ende ausgeliefert, sondern jede Stufe meldet ihr Bild, sobald es fertig
    ist (`job["stages"]`). GET /api/jobs/{id} gibt die bereits fertigen Stufen
    auch im Zustand "processing" zurueck. Dadurch fuellt sich die Kette in der
    Oberflaeche waehrend der Rechnung - die Pfeile sind echter Fortschritt und
    keine Animation.
"""

import base64
import io
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Wiederverwendung deiner bestehenden Module
from torchvision import transforms as T

import stages as pipeline_stages
from model.model import HEAD_GRID, ClassifierView, build_two_head_model

# --------------------------------------------------------------------------
# Konfiguration (per Umgebungsvariable überschreibbar)
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    """Pfad relativ zum Arbeitsverzeichnis ODER zum Repo suchen.

    Im Container liegt alles flach neben dieser Datei (`/app`), in der
    Entwicklung wird der Server aus `serving/` gestartet, waehrend die Ordner
    `checkpoints/` und `data/` eine Ebene hoeher liegen. Beide Faelle bedienen,
    statt eine Umgebungsvariable zu verlangen, deren Vergessen als
    `FileNotFoundError` beim Start endet. Absolute Pfade bleiben unangetastet.
    """
    if os.path.isabs(path):
        return path
    for base in ("", _HERE, os.path.dirname(_HERE)):
        candidate = os.path.join(base, path) if base else path
        if os.path.exists(candidate):
            return candidate
    return path                                    # Fehlermeldung nennt das Original


# --------------------------------------------------------------------------
# Die Kalibrierdatei ist die einzige Quelle fuer Gewichte, Kurven und Schwelle
# --------------------------------------------------------------------------
# Erzeugt von rsna/befunde/rsna_platt.py auf den 22872 Entwicklungsbildern, VOR
# dem Holdout. Sie nennt drei Dinge, und alle drei gehoeren zusammen:
#   * welche fuenf Gewichte das Ensemble bilden,
#   * die Platt-Kurve (a, b) je Fold,
#   * die Entscheidungsschwelle 0,2003.
# Sie an EINER Stelle zu fuehren ist die Lehre aus dem 09.08.2026: im
# Dockerfile stand ein `ENV THRESHOLD=0.5` und hat die Schwelle aus dem
# Quelltext still ueberschrieben. Im Protokoll stand eine Zahl, im Container
# lief eine andere, und nichts in der Ausgabe hat widersprochen.
ARM_TAG = "_p5head_ex"
CALIBRATION_PATH = _resolve(os.getenv("CALIBRATION_PATH",
                                      "model/kalibrierung_p10.json"))

# Ein gesetztes THRESHOLD ist ab jetzt ein Startfehler und keine Einstellung
# mehr. Lieber laut abbrechen als leise mit einer Schwelle rechnen, die zu
# keinem der fuenf Gewichte gehoert. docker-compose.yml hat genau diese
# Variable bis zum 09.08.2026 noch auf 0.5 gesetzt.
if os.getenv("THRESHOLD") is not None:
    raise RuntimeError(
        "THRESHOLD ist gesetzt. Die Schwelle gehoert zu den Gewichten und "
        "kommt ausschliesslich aus der Kalibrierdatei "
        f"({CALIBRATION_PATH}). Die Umgebungsvariable bitte entfernen; sie hat "
        "am 09.08.2026 schon einmal still eine falsche Schwelle ausgeliefert."
    )

# Der Familienschalter ist keine Einstellung mehr. Ausgeliefert wird genau ein
# Modell, das Phase-10-Ensemble; die alte Kermany-Strecke hat weder diesen Kopf
# noch diese Vorverarbeitung. Er wird nur noch gelesen, um einen alten,
# widersprechenden Wert laut abzulehnen statt ihn zu ignorieren.
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "rsna").lower()
if MODEL_FAMILY != "rsna":
    raise RuntimeError(
        f"MODEL_FAMILY={MODEL_FAMILY!r}. Seit Phase 10 liefert dieser Dienst "
        "ausschliesslich das RSNA-Ensemble aus; eine andere Familie hat weder "
        "den zweiten Kopf noch dessen Vorverarbeitung."
    )

INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224"))       # rsna_train.py --size, Vorgabe 224
TEST_DIR = _resolve(os.getenv("TEST_DIR", "data/chest_xray/test"))
CLASSES = ["NORMAL", "PNEUMONIA"]
SAMPLES_PER_CLASS = int(os.getenv("SAMPLES_PER_CLASS", "4"))

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))         # max. Analysen pro Fenster
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "3600"))     # Fensterlänge in Sekunden
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
JOB_TTL = int(os.getenv("JOB_TTL", "300"))              # fertige Jobs nach 5 min vergessen
ALLOWED_TYPES = {"image/png", "image/jpeg"}
SHOW_STAGES = os.getenv("SHOW_STAGES", "1") not in ("0", "false", "False")

torch.set_num_threads(1)  # begrenzt CPU/RAM auf schwacher Hardware

# --------------------------------------------------------------------------
# Kalibrierung lesen, fuenf Modelle + fuenf Grad-CAMs einmalig laden
# --------------------------------------------------------------------------
if not os.path.isfile(CALIBRATION_PATH):
    raise RuntimeError(
        f"{CALIBRATION_PATH} fehlt. Ohne sie gaebe es keine Wahrscheinlich"
        "keiten, sondern nur Rohwerte, und die Schwelle waere bedeutungslos. "
        "Die Datei gehoert ins Repo und in das Container-Abbild."
    )
with open(CALIBRATION_PATH, encoding="utf-8") as _fh:
    CALIBRATION = json.load(_fh)

if CALIBRATION.get("arm") != ARM_TAG:
    raise RuntimeError(
        f"die Kalibrierdatei gehoert zu Arm {CALIBRATION.get('arm')!r}, "
        f"gebraucht wird {ARM_TAG!r}."
    )

PLATT = {int(e["fold"]): (float(e["a"]), float(e["b"])) for e in CALIBRATION["platt"]}
FOLDS = sorted(PLATT)
CHECKPOINT_PATHS = [_resolve(p) for p in CALIBRATION["checkpoints"]]
if len(CHECKPOINT_PATHS) != len(FOLDS):
    raise RuntimeError(
        f"die Kalibrierdatei nennt {len(CHECKPOINT_PATHS)} Gewichte, aber "
        f"{len(FOLDS)} Kurven. Das Ensemble sind ALLE Folds, nicht eine "
        f"Auswahl daraus."
    )
THRESHOLD = float(CALIBRATION["schwelle"])

print(f"Lade Ensemble aus {len(FOLDS)} Gewichten (Arm {ARM_TAG}) ...")
NETS: list[torch.nn.Module] = []
CAMS: list[GradCAM] = []
for _k, _path in zip(FOLDS, CHECKPOINT_PATHS):
    # Die Kurve wird ueber die REIHENFOLGE der beiden Listen an das Gewicht
    # gebunden, und eine vertauschte Reihenfolge waere von aussen unsichtbar:
    # herauskaeme eine Wahrscheinlichkeit, die wie eine aussieht. Deshalb muss
    # der Dateiname die Fold-Nummer nennen, zu der die Kurve gehoert.
    if f"_f{_k}_" not in os.path.basename(_path):
        raise RuntimeError(
            f"Fold {_k} bekaeme die Platt-Kurve von {os.path.basename(_path)}. "
            f"Reihenfolge der Listen 'checkpoints' und 'platt' in "
            f"{CALIBRATION_PATH} pruefen."
        )
    if not os.path.isfile(_path):
        raise RuntimeError(
            f"{_path} fehlt. Das ausgelieferte Modell ist das Mittel ueber "
            f"ALLE {len(FOLDS)} Folds; mit vier davon waere es ein anderes "
            f"Modell als das auf dem Holdout gemessene."
        )
    # weights_only=True: die Datei enthaelt nur Gewichte, also nicht den ganzen
    # Pickle-Interpreter zulassen. Ab Torch 2.6 ist das ohnehin die Vorgabe.
    _state = torch.load(_path, map_location="cpu", weights_only=True)
    _net = build_two_head_model(HEAD_GRID)
    # Dieselbe Pruefung wie in rsna/pipeline/rsna_holdout.py: ein einkoepfiges
    # Gewicht wuerde bei strict=False klaglos laden und stillschweigend ein
    # anderes Modell ausliefern. Erst der Abgleich der Schluessel faengt das.
    _missing, _unexpected = _net.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        raise RuntimeError(
            f"{os.path.basename(_path)} passt nicht auf das zweikoepfige "
            f"Modell. Fehlend {list(_missing)[:4]}, ueberzaehlig "
            f"{list(_unexpected)[:4]}."
        )
    _net.eval()
    NETS.append(_net)
    # ClassifierView reicht den Logit heraus und stellt DASSELBE layer4-Objekt
    # nach aussen, auf das Grad-CAM beim einkoepfigen Modell gezeigt hat. Das
    # zweikoepfige Modell gaebe ein Tupel zurueck, und daran bricht
    # pytorch_grad_cam.
    _view_net = ClassifierView(_net)
    CAMS.append(GradCAM(model=_view_net, target_layers=[_view_net.layer4]))
    print(f"  Fold {_k}: {os.path.basename(_path)}  "
          f"Platt a={PLATT[_k][0]:.4f} b={PLATT[_k][1]:.4f}")
print(f"Ensemble geladen. Schwelle {THRESHOLD:.4f} aus {CALIBRATION_PATH}.")


# --------------------------------------------------------------------------
# Vorverarbeitung: MUSS zum geladenen Gewicht passen
# --------------------------------------------------------------------------
# Nachgebaut aus rsna/pipeline/rsna_train.py, build_transforms(size, train=False).
# Bewusst nicht importiert: das Trainingsmodul zieht die halbe Trainingsstrecke
# mit, und der Serving-Prozess soll klein bleiben. Wer dort die Transform
# aendert, muss sie hier mitziehen; deshalb steht die Quelle im Kommentar.
#
# ACHTUNG, GEAENDERT BEIM UMBAU AUF DAS ENSEMBLE: das Bild wird jetzt ZUERST
# nach Graustufen gewandelt und DANN skaliert. Der Trainingslader tut genau das
# (`Image.open(...).convert("L")` in RsnaDataset, danach `Resize`), die App hat
# bis dahin ein RGB-Bild skaliert und erst danach entfaerbt. Bei einer grauen
# Roentgenaufnahme ist der Unterschied null, bei einem farbigen Upload nicht.
# Der Rauchtest in tests/test_serving_ensemble.py vergleicht die App gegen
# rsna_holdout.py, und ohne diese Reihenfolge waere er nicht zu bestehen
# gewesen.
_IMNET_MEAN, _IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
rsna_transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(_IMNET_MEAN, _IMNET_STD),
])


def model_input(pil_img: Image.Image) -> torch.Tensor:
    """Ein PIL-Bild als Modelleingabe (3, 224, 224), ohne Stapel-Achse."""
    return rsna_transform(pil_img.convert("L"))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt_apply(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Die Platt-Kurve eines Folds, Zeichen fuer Zeichen wie im Holdout.

    Quelle: rsna/pipeline/rsna_holdout.py, `platt_apply`. Dieselbe Formel und
    dasselbe eps, damit die App fuer dasselbe Bild dieselbe Zahl liefert wie
    die Auswertung. Der Rauchtest prueft genau das nach.
    """
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))

# Der Segmenter ist OPTIONAL. Fehlt checkpoints/unet_best.pth, laeuft die
# Analyse unveraendert weiter, nur die Lungenfinder-Karte entfaellt. Sie war nie
# im Wertungspfad, also fehlt dem Score nichts.
if SHOW_STAGES:
    if pipeline_stages.load_segmenter():
        print("U-Net (Lungensegmentierung) geladen.")
    else:
        print(f"U-Net nicht verfuegbar ({pipeline_stages.segmenter_status()['error']}) - "
              f"Lungenfinder wird uebersprungen.")


def _now_iso(ts: float) -> str:
    """Unix-Zeit -> ISO-8601 in UTC, z.B. 2026-07-23T18:30:00Z."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _png_base64(rgb_uint8: np.ndarray) -> str:
    """Ein RGB-uint8-Array als base64-kodiertes PNG (ohne data:-Präfix)."""
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Unsicherheit durch Test-Time-Augmentation
# --------------------------------------------------------------------------
# Warum ausgerechnet Ausschnitts-Verschiebungen und keine Drehungen oder
# Helligkeitsaenderungen:
#   * Helligkeit taugt nicht als Sonde. Gerechnet wird mit fester
#     ImageNet-Normierung, dort WAERE Helligkeit wirksam - aber das Training
#     augmentiert bereits mit ColorJitter(0.15), eine Helligkeitssonde misst
#     also die Augmentierung und nicht das Bild. Phase 9 hat die Sache
#     nachgemessen: der Kanal, an dem die Aufnahmeart haengt, ist ohnehin der
#     Kontrast und nicht die Helligkeit.
#   * Drehungen erzeugen schwarze Ecken. Deren Wirkung auf den Score waere ein
#     Artefakt der Fuellfarbe, nicht der Anatomie.
#   * AUSSCHNITT und ZOOM sind dagegen genau der Kanal, der in diesem Projekt
#     als Confounder nachgewiesen wurde. Die Frage "wie weit wandert der Score,
#     wenn der Bildausschnitt um zwei Prozent wandert" misst also die
#     Empfindlichkeit dort, wo sie nachweislich weh tut.
# Das ist AUSDRUECKLICH kein Konfidenzintervall: es misst die Streuung des
# Modells gegen kleine Bildaenderungen, nicht die Unsicherheit der Diagnose.
# Die Oberflaeche beschriftet es genau so.
TTA_VIEWS = [
    ("full frame", 1.00, 0.0, 0.0),
    ("96 % crop", 0.96, 0.0, 0.0),
    ("92 % crop", 0.92, 0.0, 0.0),
    ("shifted right/down", 0.96, 0.02, 0.02),
    ("shifted left/up", 0.96, -0.02, -0.02),
]


def _view(pil_img: Image.Image, frac: float, dx: float, dy: float) -> Image.Image:
    """Mittiger Ausschnitt der Groesse `frac`, um (dx, dy) verschoben.

    Verschiebung in Anteilen der Bildkante. Der Ausschnitt wird an den Rand
    geklemmt, damit nie ueber das Bild hinaus geschnitten wird - sonst haette
    die Variante einen schwarzen Streifen und maesse wieder ein Artefakt.
    """
    if frac >= 1.0 and dx == 0.0 and dy == 0.0:
        return pil_img
    W, H = pil_img.size
    w, h = int(W * frac), int(H * frac)
    left = min(max(int((W - w) / 2 + dx * W), 0), W - w)
    top = min(max(int((H - h) / 2 + dy * H), 0), H - h)
    return pil_img.crop((left, top, left + w, top + h))


@torch.no_grad()
def ensemble_scores(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray, list[float], torch.Tensor]:
    """Die ganze Rechnung des ausgelieferten Modells in einem Durchgang.

    Rueckgabe:
      p_views    kalibrierte Ensemble-Wahrscheinlichkeit je TTA-Variante.
                 `p_views[0]` gehoert zum unveraenderten Bild und ist DIE Zahl,
                 die die Oberflaeche anzeigt.
      feld       das gemittelte Kopffeld des unveraenderten Bildes, 14 x 14,
                 nach der Sigmoid-Funktion. Genau wie in rsna_holdout.py.
      p_folds    die fuenf Einzelwahrscheinlichkeiten des unveraenderten
                 Bildes, kalibriert. Nur zur Anzeige, sie gehen als Mittel
                 ohnehin in p_views[0] ein.
      x0         die Modelleingabe des unveraenderten Bildes, fuer Grad-CAM und
                 fuer die Stufenanzeige.

    Die fuenf Varianten gehen als EIN Stapel durch jedes Modell, nicht einzeln.
    Das sind fuenf Vorwaertsrechnungen statt fuenfundzwanzig; auf einem Kern
    ist das der Unterschied zwischen einer und mehreren Sekunden.

    Die Reihenfolge ist die des Holdout-Skripts und nicht umstellbar: erst
    Sigmoid, dann Platt JE FOLD, dann mitteln. Wer erst mittelt und danach
    kalibriert, rechnet etwas anderes aus.
    """
    batch = torch.stack([model_input(_view(pil_img, frac, dx, dy))
                         for _, frac, dx, dy in TTA_VIEWS])

    summe = np.zeros(len(TTA_VIEWS), dtype=float)
    summe_feld = np.zeros((HEAD_GRID, HEAD_GRID), dtype=float)
    p_folds: list[float] = []

    for k, net in zip(FOLDS, NETS):
        logit, feld = net(batch)
        p_roh = torch.sigmoid(logit[:, 0]).numpy().astype(float)
        a, b = PLATT[k]
        p_kal = platt_apply(p_roh, a, b)
        summe += p_kal
        # Nur das unveraenderte Bild traegt zum Kopffeld bei. Ein Feld ueber
        # verschobene Ausschnitte zu mitteln waere ein verwischtes Feld, und
        # verwischt heisst hier nicht vorsichtig, sondern falsch verortet.
        summe_feld += torch.sigmoid(feld[0, 0]).numpy().astype(float)
        p_folds.append(float(p_kal[0]))

    return (summe / len(FOLDS), summe_feld / len(FOLDS), p_folds,
            batch[0:1])


def ensemble_cam(input_tensor: torch.Tensor) -> np.ndarray:
    """Grad-CAM des Ensembles: der Mittelwert der fuenf Einzelkarten.

    Die Karte EINES Folds zu zeigen waere die Erklaerung eines Modells, das die
    angezeigte Zahl nicht erzeugt hat. Jede Einzelkarte ist von
    `pytorch_grad_cam` bereits auf 0 bis 1 gestreckt, das Mittel ist also der
    Anteil der Folds, die eine Stelle warm finden, und nicht die Summe
    unvergleichbarer Skalen.

    Kostet fuenf Rueckwaertsschritte statt einem. Das ist der Preis dafuer,
    dass die Karte zu der Zahl daneben gehoert.
    """
    acc = None
    for c in CAMS:
        g = c(input_tensor=input_tensor)[0]
        acc = g if acc is None else acc + g
    return acc / len(CAMS)


def run_inference(pil_img: Image.Image, emit=None) -> dict:
    """Klassifikation + Grad-CAM für ein Bild. Läuft NUR im Worker-Thread.

    `emit(stage_dict)` wird nach JEDER fertigen Stufe aufgerufen, nicht erst am
    Ende. Der Aufrufer haengt die Stufe unter Lock an den Job an, sodass ein
    parallel laufendes GET /api/jobs/{id} sie sofort sieht.

    Wertungspfad (stages.PIPELINE), und nur diese drei sind eine Kette:
        1. Hochgeladenes Bild
        2. Modelleingabe 224x224 (Graustufen, skalieren, ImageNet-Normierung)
        3. Grad-CAM, gemittelt ueber die fuenf Modelle

    Daneben (stages.ASIDES), ausdruecklich KEINE Kettenglieder:
        Lungenmaske (U-Net), `group: "aside"`
        Kopffeld des Ensembles, `group: "aside"`

    Klassifiziert wird das Vollbild, genau so wie trainiert wurde: dieselbe
    Skalierung und dieselbe feste Normierung wie in
    `rsna_train.build_transforms(224, train=False)`, weder Maske noch
    Zuschnitt. Die Maske wird echt gerechnet und gezeigt, beruehrt den Score
    aber nicht.

    Die Zuschnitt-Stufe ist entfallen. Sie stand zwischen Maske und
    Modelleingabe und legte damit nahe, der Zuschnitt werde klassifiziert; genau
    diese Lesart ist beim Lesen der Oberflaeche aufgetreten. `render_crop` in
    stages.py bleibt fuer die Messungen erhalten, wird hier aber nicht gerufen.
    """
    t0 = time.time()

    def _emit(*args, **kwargs):
        if emit is not None:
            emit(pipeline_stages.stage(*args, **kwargs))

    def _stage_ms(started):
        return int((time.time() - started) * 1000)

    if emit is not None:
        ts = time.time()
        _emit("upload", pipeline_stages.render_original(pil_img),
              ms=_stage_ms(ts), size=list(pil_img.size))

        # --- Nebenschauplatz: Lungenmaske ---------------------------------
        # Traegt `group: "aside"` und wird von der Oberflaeche in einer eigenen
        # Karte gezeigt, nicht als Glied der Kette. Faellt sie aus, laeuft der
        # Wertungspfad unveraendert weiter.
        mask = None
        ts = time.time()
        try:
            mask = pipeline_stages.lung_mask(pil_img)
        except Exception as exc:  # noqa: BLE001
            _emit("mask", None, skipped=True, reason=f"{type(exc).__name__}: {exc}")
        if mask is not None:
            _emit("mask", pipeline_stages.render_mask_overlay(pil_img, mask),
                  ms=_stage_ms(ts))
        elif pipeline_stages.segmenter_status()["available"]:
            _emit("mask", None, skipped=True, reason="No lung region found in this image.")
        else:
            _emit("mask", None, skipped=True, reason="Segmentation model not available.")

    # --- 2: Modelleingabe + die ganze Ensemble-Rechnung --------------------
    # Beides in einem Schritt, weil die Modelleingabe des unveraenderten Bildes
    # ohnehin als erste Zeile des TTA-Stapels entsteht. Sie ein zweites Mal zu
    # bauen hiesse, eine Zahl aus einem anderen Rechenweg anzuzeigen als die,
    # die gerechnet wurde.
    ts = time.time()
    p_views, feld, p_folds, input_tensor = ensemble_scores(pil_img)
    if emit is not None:
        _emit("model_input", pipeline_stages.render_model_input(input_tensor),
              ms=_stage_ms(ts))

    prob_pneu = float(p_views[0])
    views = [{"view": name, "probability": round(float(p), 4)}
             for (name, _, _, _), p in zip(TTA_VIEWS, p_views)]
    vals = sorted(v["probability"] for v in views)
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    # --- 3: Grad-CAM (braucht Gradienten, daher kein no_grad) --------------
    ts = time.time()
    grayscale_cam = ensemble_cam(input_tensor)
    rgb = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 RGB
    heatmap_b64 = _png_base64(overlay)
    if emit is not None:
        _emit("heatmap", heatmap_b64, ms=_stage_ms(ts))

    # --- daneben: das Kopffeld ---------------------------------------------
    # Der zweite Ausgang desselben Modells, also KEIN Nebenschauplatz im Sinne
    # der Lungenmaske: er stammt aus genau dem Netz, das die Zahl oben erzeugt
    # hat. Er beruehrt die Zahl trotzdem nicht, deshalb steht er neben der
    # Kette und nicht darin.
    #
    # Ohne jede Schwelle gezeichnet, und das ist keine Bequemlichkeit: der
    # Pegel des Kopfes ist unkalibriert, auf gesunden Bildern schlaegt er in
    # 62 Prozent der Faelle an (Phase 5b). Ein Kasten oder eine Umrandung
    # waere eine Behauptung ueber genau die Groesse, die nachweislich nicht
    # stimmt. Ein Verlauf ist die staerkste Aussage, die die Messung traegt.
    head_b64 = None
    if emit is not None:
        ts = time.time()
        head_b64 = pipeline_stages.render_head_field(pil_img, feld)
        _emit("head_field", head_b64, ms=_stage_ms(ts))

    # Ohne Labelfeld, mit Absicht: die Entscheidungsschwelle traegt zwischen
    # Datensaetzen nicht (NPV 0.500 in der externen Pruefung, siehe README
    # Abschnitt 6). Ein Label waere genau die Aussage, die dort gescheitert
    # ist. Wer eines ergaenzen will, muss vorher die Kalibrierung zeigen.
    return {
        "probability": round(prob_pneu, 4),
        "threshold": THRESHOLD,
        "uncertainty": {
            "median": round(median, 4),
            "min": min(vals),
            "max": max(vals),
            "spread": round(max(vals) - min(vals), 4),
            "views": views,
            "method": "test-time augmentation over small framing changes",
        },
        # Die zweite Streuungsquelle, und die interessantere: nicht wie weit der
        # Wert unter Bildaenderungen wandert, sondern wie weit die fuenf
        # Modelle auseinanderliegen, aus denen der angezeigte Mittelwert
        # gebildet ist.
        "ensemble": {
            "folds": len(FOLDS),
            "per_fold": [round(p, 4) for p in p_folds],
            "spread": round(max(p_folds) - min(p_folds), 4),
        },
        "head_field": {
            "grid": HEAD_GRID,
            "max": round(float(feld.max()), 4),
            "values": [[round(float(v), 4) for v in row] for row in feld],
        },
        "head_field_png_base64": head_b64,
        "heatmap_png_base64": heatmap_b64,
        "inference_ms": int((time.time() - t0) * 1000),
    }


# --------------------------------------------------------------------------
# Beispielbilder registrieren (feste Auswahl aus dem Test-Ordner)
# --------------------------------------------------------------------------
def _build_sample_registry() -> dict:
    registry = {}
    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for i, fname in enumerate(files[:SAMPLES_PER_CLASS], start=1):
            sid = f"{cls.lower()}_{i:02d}"
            path = os.path.join(folder, fname)
            # Thumbnail einmalig erzeugen und als PNG-Bytes vorhalten (nur wenige Bilder)
            img = Image.open(path).convert("RGB")
            img.thumbnail((160, 160))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            registry[sid] = {
                "id": sid,
                "label": f"{cls.title()} X-ray #{i}",
                "category": cls,
                "path": path,
                "thumb_png": buf.getvalue(),
            }
    return registry


SAMPLES = _build_sample_registry()
print(f"{len(SAMPLES)} Beispielbilder registriert.")

# --------------------------------------------------------------------------
# Job-Verwaltung + Warteschlange
# --------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_queue: "queue.Queue[str]" = queue.Queue()
_seq_counter = 0


def _next_seq() -> int:
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


def _position_of(job: dict) -> int:
    """Wie viele Jobs müssen vor diesem noch laufen? (1 = als nächstes dran)."""
    ahead = 0
    for j in jobs.values():
        if j["status"] == "processing":
            ahead += 1
        elif j["status"] == "queued" and j["seq"] < job["seq"]:
            ahead += 1
    return ahead + 1


def _purge_old_jobs() -> None:
    """Fertige/fehlerhafte Jobs nach JOB_TTL entfernen (spart RAM)."""
    now = time.time()
    dead = [
        jid for jid, j in jobs.items()
        if j["status"] in ("done", "error") and now - j.get("finished_at", now) > JOB_TTL
    ]
    for jid in dead:
        jobs.pop(jid, None)


def _worker() -> None:
    """Endlosschleife: nimmt einen Job aus der Queue und verarbeitet ihn."""
    while True:
        job_id = work_queue.get()
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:            # inzwischen entfernt
                work_queue.task_done()
                continue
            job["status"] = "processing"
            img = job.pop("image")     # Bild aus dem Job herausnehmen und verarbeiten

        def publish(stage: dict, _job=job) -> None:
            """Eine fertige Stufe an den Job haengen, damit sie sofort abrufbar
            ist. Kurz gesperrt, weil GET /api/jobs parallel liest."""
            with jobs_lock:
                _job.setdefault("stages", []).append(stage)

        try:
            result = run_inference(img, emit=publish if SHOW_STAGES else None)
            with jobs_lock:
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            with jobs_lock:
                job["status"] = "error"
                job["error"] = "inference_failed"
                job["message"] = str(exc)
                job["finished_at"] = time.time()
        finally:
            work_queue.task_done()


# Genau EIN Worker-Thread => immer nur eine Inferenz gleichzeitig
threading.Thread(target=_worker, daemon=True).start()

# --------------------------------------------------------------------------
# Rate-Limiting (fixes Zeitfenster pro Client)
# --------------------------------------------------------------------------
_rl: dict[str, dict] = {}
_rl_lock = threading.Lock()


def _rl_entry(client_id: str, now: float) -> dict:
    e = _rl.get(client_id)
    if e is None or now - e["start"] >= RATE_WINDOW:
        e = {"count": 0, "start": now}
        _rl[client_id] = e
    return e


def limits_for(client_id: str) -> dict:
    now = time.time()
    with _rl_lock:
        e = _rl_entry(client_id, now)
        remaining = max(0, RATE_LIMIT - e["count"])
        reset_at = e["start"] + RATE_WINDOW
    return {
        "limit": RATE_LIMIT,
        "remaining": remaining,
        "window_seconds": RATE_WINDOW,
        "reset_at": _now_iso(reset_at),
    }


def consume_quota(client_id: str):
    """Verbraucht 1 Analyse. Gibt (ok, reset_at_iso) zurück."""
    now = time.time()
    with _rl_lock:
        e = _rl_entry(client_id, now)
        if e["count"] >= RATE_LIMIT:
            return False, _now_iso(e["start"] + RATE_WINDOW)
        e["count"] += 1
        return True, _now_iso(e["start"] + RATE_WINDOW)


# --------------------------------------------------------------------------
# FastAPI-App
# --------------------------------------------------------------------------
app = FastAPI(title="Pneumonia Detection API")

# Im Dev-Betrieb läuft die UI auf :5173 (andere Origin) -> CORS erlauben.
# In Produktion liegt alles hinter nginx (gleiche Origin), dann ist das egal.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_id(x_client_id: str | None) -> str:
    return x_client_id or "anonymous"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "stages_enabled": SHOW_STAGES,
        "segmentation_loaded": pipeline_stages.segmenter_status()["available"],
        # Damit im Betrieb nachzusehen ist, WAS geladen wurde, ohne in die
        # Protokolle zu greifen. Die Schwelle steht bewusst dabei: sie ist die
        # Zahl, die am 09.08.2026 im Container eine andere war als im Text.
        "ensemble": {
            "arm": ARM_TAG,
            "folds": len(FOLDS),
            "threshold": round(THRESHOLD, 4),
            "calibration": os.path.basename(CALIBRATION_PATH),
        },
    }


@app.get("/api/limits")
def get_limits(x_client_id: str | None = Header(default=None)):
    return limits_for(_client_id(x_client_id))


@app.get("/api/samples")
def get_samples():
    return {
        "samples": [
            {
                "id": s["id"],
                "label": s["label"],
                "category": s["category"],
                "thumbnail_url": f"/api/samples/{s['id']}/thumb",
            }
            for s in SAMPLES.values()
        ]
    }


@app.get("/api/samples/{sample_id}/thumb")
def sample_thumb(sample_id: str):
    s = SAMPLES.get(sample_id)
    if s is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return Response(content=s["thumb_png"], media_type="image/png")


def _enqueue(image: Image.Image) -> dict:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        _purge_old_jobs()
        job = {"job_id": job_id, "status": "queued", "seq": _next_seq(), "image": image}
        jobs[job_id] = job
        position = _position_of(job)
    work_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "position": position}


@app.post("/api/analyze")
async def analyze(request: Request, x_client_id: str | None = Header(default=None)):
    client = _client_id(x_client_id)

    # 1) Rate-Limit prüfen und verbrauchen
    ok, reset_at = consume_quota(client)
    if not ok:
        return JSONResponse(
            {"error": "rate_limited", "message": "Analysis limit reached.", "reset_at": reset_at},
            status_code=429,
        )

    content_type = request.headers.get("content-type", "")

    # 2a) Datei-Upload (multipart/form-data)
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            return JSONResponse({"error": "invalid_request", "message": "No file field."}, status_code=400)
        if upload.content_type not in ALLOWED_TYPES:
            return JSONResponse({"error": "unsupported_type", "message": "Use PNG or JPEG."}, status_code=415)
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "file_too_large", "message": "Max 10 MB."}, status_code=413)
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid_request", "message": "Not a valid image."}, status_code=400)
        return JSONResponse(_enqueue(image), status_code=202)

    # 2b) Server-Beispielbild ({"sample_id": "..."})
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid_request", "message": "Expected JSON or file."}, status_code=400)
    sample_id = (body or {}).get("sample_id")
    s = SAMPLES.get(sample_id)
    if s is None:
        return JSONResponse({"error": "invalid_request", "message": "Unknown sample_id."}, status_code=400)
    image = Image.open(s["path"]).convert("RGB")
    return JSONResponse(_enqueue(image), status_code=202)


@app.get("/api/pipeline")
def get_pipeline():
    """Die Kette ohne Bilder: Titel, Beschriftung und Status je Stufe.

    Damit kann die Oberflaeche die leere Kette samt Pfeilen schon zeichnen,
    bevor ueberhaupt ein Bild hochgeladen wurde.

    `stages` ist ausschliesslich der Wertungspfad, also genau die Stufen,
    zwischen denen ein Pfeil zu Recht steht. `asides` sind Rechnungen, die auf
    demselben Bild laufen und gezeigt werden, aber den Score nicht beruehren;
    sie gehoeren in eine eigene Karte und nicht in die Reihe. `status:
    "explored"` markiert dort weiterhin, was untersucht und verworfen wurde.
    """
    return {
        "stages": pipeline_stages.PIPELINE,
        "asides": pipeline_stages.ASIDES,
        "enabled": SHOW_STAGES,
        "segmentation": pipeline_stages.segmenter_status(),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, since: int = 0):
    """`since` = Anzahl der Stufen, die der Client schon hat.

    Ohne diesen Parameter kaeme bei jedem Poll die komplette Bilderkette erneut
    ueber die Leitung. Der Client zaehlt mit und bekommt nur den Zuwachs.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "job_not_found"}, status_code=404)

        status = job["status"]
        all_stages = job.get("stages", [])
        since = max(0, min(int(since), len(all_stages)))
        new_stages = list(all_stages[since:])
        stage_info = {"stages": new_stages, "stages_total": len(all_stages)}

        if status == "queued":
            return {"job_id": job_id, "status": "queued", "position": _position_of(job),
                    **stage_info}
        if status == "processing":
            return {"job_id": job_id, "status": "processing", **stage_info}
        if status == "done":
            return {"job_id": job_id, "status": "done", "result": job["result"], **stage_info}
        # error
        return {
            "job_id": job_id,
            "status": "error",
            "error": job.get("error", "inference_failed"),
            "message": job.get("message", "Analysis failed."),
            **stage_info,
        }


if __name__ == "__main__":
    # Lokaler Start:  python main.py   (oder: uvicorn main:app --reload)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Die sechs Demobilder der Webapp nach fester Regel aus dem Holdout ziehen.

Die Regel steht in `erklaerungen/31_webapp_karten_und_skala.md`, Abschnitt 4,
und sie stand dort, BEVOR ein Kandidat angesehen wurde:

    Grundmenge  die 3812 Holdout-Bilder
    Einteilung  RSNA-Klasse (3) mal Aufnahmeart (2) = sechs Zellen
    Auswahl     je Zelle das Bild am MEDIAN der Ensemble-Wahrscheinlichkeit
    Gleichstand die lexikografisch kleinste patientId
    kein Tausch danach

Warum der Median und nicht der beste Fall: ein Demobild soll zeigen, was das
Modell ueblicherweise tut. Nach dem Ergebnis auszuwaehlen waere genau der Griff,
gegen den sich dieses Projekt neun Phasen lang mit Vorfestlegungen gewehrt hat.

Was hier NICHT passiert: es wird keine Kennzahl gebildet und keine berichtete
Zahl beruehrt. Der Holdout ist als MESSGROESSE verbraucht; hier werden Bilder
ausgesucht.

Ausgabe:
    serving/samples/<id>.png        sechs Bilder
    serving/samples/manifest.json   was sie sind, woher sie kommen

Aufruf:  python rsna/befunde/rsna_demobilder.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

import _repo_path  # noqa: F401  (setzt sys.path auf die Repo-Wurzel)

WURZEL = Path(__file__).resolve().parents[2]
HOLDOUT = WURZEL / "predictions_holdout" / "holdout.csv"
KLASSEN = WURZEL / "data" / "rsna" / "stage_2_detailed_class_info.csv"
PNG_DIR = WURZEL / "data" / "rsna" / "png512"
ZIEL = WURZEL / "serving" / "samples"

SOLL_N = 3812                       # Holdout-Groesse, Phase 10

# RSNA-Klasse -> (Schluessel, Anzeigename). Die Reihenfolge ist die
# Anzeigereihenfolge in der Oberflaeche.
KLASSEN_MAP = [
    ("Normal", "normal", "Normal"),
    ("No Lung Opacity / Not Normal", "not_normal", "Abnormal, no lung opacity"),
    ("Lung Opacity", "opacity", "Lung opacity (pneumonia)"),
]
VIEWS = ["AP", "PA"]


def abbruch(text: str) -> None:
    print(f"ABBRUCH: {text}", file=sys.stderr)
    raise SystemExit(1)


def pruefsumme(pfad: Path) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    for pfad in (HOLDOUT, KLASSEN):
        if not pfad.is_file():
            abbruch(f"{pfad} fehlt.")

    hold = pd.read_csv(HOLDOUT)
    if len(hold) != SOLL_N:
        abbruch(f"{HOLDOUT.name} hat {len(hold)} Zeilen, erwartet {SOLL_N}.")
    for spalte in ("patientId", "y", "viewpos", "p_ens"):
        if spalte not in hold.columns:
            abbruch(f"Spalte {spalte!r} fehlt in {HOLDOUT.name}.")

    # Eine Zeile JE KASTEN, also mehrere je Bild. Die Klasse ist je Bild
    # konstant; das wird geprueft statt angenommen, weil ein stiller
    # drop_duplicates hier eine falsche Klasse einziehen koennte.
    kl = pd.read_csv(KLASSEN)
    n_klassen = kl.groupby("patientId")["class"].nunique()
    if (n_klassen > 1).any():
        abbruch("mindestens eine patientId traegt zwei verschiedene Klassen.")
    kl = kl.drop_duplicates("patientId")[["patientId", "class"]]

    df = hold.merge(kl, on="patientId", how="left")
    if df["class"].isna().any():
        abbruch(f"{int(df['class'].isna().sum())} Holdout-Bilder ohne Klasse.")

    # Gegenprobe: die RSNA-Klasse und das Trainingslabel muessen zueinander
    # passen. y == 1 genau fuer "Lung Opacity". Waere das verletzt, zeigte die
    # Demo eine Wahrheit an, die nicht die gemessene ist.
    passt = (df["class"] == "Lung Opacity") == (df["y"] == 1.0)
    if not passt.all():
        abbruch(f"{int((~passt).sum())} Bilder: Klasse und Label widersprechen sich.")

    unbekannt = set(df["viewpos"]) - set(VIEWS)
    if unbekannt:
        abbruch(f"unbekannte Aufnahmeart(en): {sorted(unbekannt)}")

    ZIEL.mkdir(parents=True, exist_ok=True)
    eintraege = []
    print(f"Grundmenge {len(df)} Bilder aus {HOLDOUT.name} "
          f"(sha256 {pruefsumme(HOLDOUT)})\n")
    print(f"{'Zelle':<34} {'n':>5} {'Median':>8} {'gewaehlt':>9}  patientId")
    print("-" * 100)

    for rsna_name, schluessel, anzeige in KLASSEN_MAP:
        for view in VIEWS:
            zelle = df[(df["class"] == rsna_name) & (df["viewpos"] == view)]
            if zelle.empty:
                abbruch(f"Zelle {rsna_name} / {view} ist leer.")
            median = float(zelle["p_ens"].median())
            # Abstand zum Median, Gleichstand ueber die patientId. Beides in
            # einer Sortierung, damit die Wahl reproduzierbar ist und nicht von
            # der Zeilenreihenfolge abhaengt.
            gewaehlt = (zelle.assign(_d=(zelle["p_ens"] - median).abs())
                             .sort_values(["_d", "patientId"])
                             .iloc[0])
            pid = str(gewaehlt["patientId"])
            quelle = PNG_DIR / f"{pid}.png"
            if not quelle.is_file():
                abbruch(f"{quelle} fehlt.")

            sid = f"{schluessel}_{view.lower()}"
            shutil.copyfile(quelle, ZIEL / f"{sid}.png")
            eintraege.append({
                "id": sid,
                "file": f"{sid}.png",
                "label": f"{anzeige}, {view}",
                "category": schluessel,
                "category_label": anzeige,
                "viewpos": view,
                "rsna_class": rsna_name,
                "y": int(gewaehlt["y"]),
                # Nur zur Herkunft, wird NICHT ueber die API ausgeliefert: eine
                # vorab bekannte Zahl neben einem Demobild machte die Demo zur
                # Behauptung statt zur Rechnung.
                "p_ens_holdout": round(float(gewaehlt["p_ens"]), 4),
                "patientId": pid,
                "zelle_n": int(len(zelle)),
                "zelle_median": round(median, 4),
            })
            print(f"{rsna_name + ' / ' + view:<34} {len(zelle):>5} {median:>8.4f} "
                  f"{float(gewaehlt['p_ens']):>9.4f}  {pid}")

    manifest = {
        "erzeugt_von": "rsna/befunde/rsna_demobilder.py",
        "regel": "erklaerungen/31_webapp_karten_und_skala.md, Abschnitt 4",
        "quelle": "predictions_holdout/holdout.csv",
        "quelle_sha256_16": pruefsumme(HOLDOUT),
        "auswahl": "je Klasse x Aufnahmeart das Bild am Median der "
                   "Ensemble-Wahrscheinlichkeit, Gleichstand nach patientId",
        "bilder": eintraege,
    }
    (ZIEL / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(eintraege)} Bilder und manifest.json geschrieben nach {ZIEL}")


if __name__ == "__main__":
    main()

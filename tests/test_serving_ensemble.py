"""Der Rauchtest fuer die ausgelieferte App.

Er beantwortet zwei Fragen, und die zweite ist die, wegen der es ihn gibt.

1. LAUFEN DIE BEIDEN FASSUNGEN VON `TwoHeadNet` AUSEINANDER?
   Die Klasse steht zweimal im Repo: einmal in `rsna/pipeline/rsna_train.py`,
   wo sie trainiert wird, und einmal in `serving/model/model.py`, wo sie
   geladen wird. Das ist eine bewusste Entscheidung vom 09.08.2026 (der
   Serving-Prozess soll die Trainingsstrecke nicht mitziehen) und es ist der
   klassische Weg, sich still ein anderes Modell auszuliefern. Der Test baut
   beide und vergleicht Parameternamen, Formen und Ausgabeformen.

2. RECHNET DIE APP DASSELBE WIE DIE AUSWERTUNG?
   `rsna/pipeline/rsna_holdout.py` hat die Zahl erzeugt, die im Ergebnis steht.
   `serving/main.py` rechnet sie im Betrieb noch einmal. Wenn die beiden
   auseinanderliegen, steht im Protokoll eine Zahl und in der App eine andere,
   und niemand merkt es. Genau dieser Fall ist in diesem Projekt schon einmal
   eingetreten, mit der Schwelle im Dockerfile.

   Der Test nimmt ein Bild aus dem Holdout, laesst die App es rechnen und
   vergleicht mit `p_ens` aus `predictions_holdout/holdout.csv`. Er
   ueberschreibt damit den Holdout nicht und rechnet ihn nicht neu; er liest
   nur die bereits geschriebene Datei.

Beide Teile ueberspringen sich selbst, wenn die noetigen Dateien nicht da sind,
damit der Test in einem frischen Klon ohne Gewichte nicht rot wird.

Aufruf (aus dem Repo-Wurzelverzeichnis, Windows):
  venv\\Scripts\\python.exe -m pytest tests\\test_serving_ensemble.py -v -s
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "serving"
PIPELINE = ROOT / "rsna" / "pipeline"
KALIBRIERUNG = SERVING / "model" / "kalibrierung_p10.json"
HOLDOUT_CSV = ROOT / "predictions_holdout" / "holdout.csv"
BILDER = ROOT / "data" / "rsna" / "png512"

# Wie eng ist eng genug. Beide Wege rechnen in float32 auf derselben CPU, aber
# nicht in derselben Stapelgroesse: der Holdout faehrt 32 Bilder auf einmal,
# die App eines. Batchnorm ist im eval-Modus davon unabhaengig, die
# Fliesskomma-Reihenfolge in den Faltungen nicht ganz. 1e-5 ist zwei
# Groessenordnungen enger als jeder Unterschied, der eine Anzeige in Prozent
# beruehren wuerde, und weit lockerer als die Rundung im Nichts.
TOLERANZ = 1e-5


def _lade(name: str, pfad: Path):
    """Ein Modul unter einem eigenen Namen laden, ohne sys.path zu vergiften."""
    spec = importlib.util.spec_from_file_location(name, pfad)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[name] = modul
    spec.loader.exec_module(modul)
    return modul


@pytest.fixture(scope="module")
def serving_model():
    """serving/model/model.py, geladen so wie main.py es sieht."""
    if str(SERVING) not in sys.path:
        sys.path.insert(0, str(SERVING))
    return _lade("serving_model_model", SERVING / "model" / "model.py")


@pytest.fixture(scope="module")
def train_modul():
    """rsna/pipeline/rsna_train.py. Zieht die Trainingsstrecke mit, was hier in
    Ordnung ist: der Test darf schwer sein, der Serving-Prozess nicht."""
    for p in (str(PIPELINE), str(ROOT / "rsna" / "befunde")):
        if p not in sys.path:
            sys.path.insert(0, p)
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    return _lade("rsna_train_fuer_test", PIPELINE / "rsna_train.py")


# --------------------------------------------------------------------------
# Teil 1: die beiden Fassungen der Klasse
# --------------------------------------------------------------------------

def test_kopfraster_gleich(serving_model, train_modul):
    """Ein anderes Raster heisst andere Gewichtsformen im Kopf."""
    assert serving_model.HEAD_GRID == train_modul.HEAD_GRID


def test_parameter_namen_und_formen_gleich(serving_model, train_modul):
    """Der eigentliche Waechter gegen das Auseinanderlaufen.

    Verglichen werden die Namen UND die Formen. Nur die Namen zu vergleichen
    liesse eine geaenderte Kanalzahl durch, nur die Formen liesse eine
    Umbenennung durch, und `load_state_dict(strict=False)` in main.py wuerde
    beides klaglos schlucken, wenn dort nicht zusaetzlich auf fehlende und
    ueberzaehlige Schluessel geprueft wuerde.
    """
    a = serving_model.TwoHeadNet(pretrained=False)
    b = train_modul.TwoHeadNet(pretrained=False)

    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}

    nur_serving = sorted(set(sa) - set(sb))
    nur_training = sorted(set(sb) - set(sa))
    assert not nur_serving, f"nur in serving/model/model.py: {nur_serving[:8]}"
    assert not nur_training, f"nur in rsna_train.py: {nur_training[:8]}"

    abweichend = {k: (sa[k], sb[k]) for k in sa if sa[k] != sb[k]}
    assert not abweichend, f"gleiche Namen, andere Formen: {abweichend}"


def test_ausgabeformen_gleich(serving_model, train_modul):
    """Beide geben (Logit, Feld) in denselben Formen zurueck."""
    x = torch.zeros(2, 3, 224, 224)
    a = serving_model.TwoHeadNet(pretrained=False).eval()
    b = train_modul.TwoHeadNet(pretrained=False).eval()
    with torch.no_grad():
        la, fa = a(x)
        lb, fb = b(x)
    assert la.shape == lb.shape == (2, 1)
    assert fa.shape == fb.shape == (2, 1, train_modul.HEAD_GRID, train_modul.HEAD_GRID)


def test_gleiche_gewichte_gleiche_zahl(serving_model, train_modul):
    """Dieselben Gewichte in beide Fassungen, dieselbe Ausgabe.

    Das ist die schaerfste der vier Pruefungen: sie faengt auch eine
    umgestellte Reihenfolge im Vorwaertsweg, die an Namen und Formen nichts
    aendert.
    """
    b = train_modul.TwoHeadNet(pretrained=False).eval()
    a = serving_model.TwoHeadNet(pretrained=False).eval()
    fehlt, zuviel = a.load_state_dict(b.state_dict(), strict=False)
    assert not fehlt and not zuviel

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        la, fa = a(x)
        lb, fb = b(x)
    assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(fa, fb, atol=1e-6)


def test_gradcam_zielt_auf_dasselbe_layer4(serving_model):
    """`ClassifierView` muss DASSELBE Objekt nach aussen stellen, nicht eine
    Kopie. pytorch_grad_cam haengt sich ueber die Objektidentitaet an."""
    net = serving_model.TwoHeadNet(pretrained=False)
    view = serving_model.ClassifierView(net)
    assert view.layer4 is net.trunk.layer4
    with torch.no_grad():
        assert view(torch.zeros(1, 3, 224, 224)).shape == (1, 1)


# --------------------------------------------------------------------------
# Teil 2: die App gegen die Auswertung
# --------------------------------------------------------------------------

def test_kalibrierdatei_vollstaendig():
    """Ohne sie startet die App nicht, und mit einer halben rechnet sie falsch."""
    if not KALIBRIERUNG.is_file():
        pytest.skip(f"{KALIBRIERUNG} fehlt")
    kal = json.loads(KALIBRIERUNG.read_text(encoding="utf-8"))
    assert kal["arm"] == "_p5head_ex"
    assert sorted(int(e["fold"]) for e in kal["platt"]) == [0, 1, 2, 3, 4]
    assert len(kal["checkpoints"]) == 5
    assert 0.0 < float(kal["schwelle"]) < 1.0


@pytest.mark.parametrize("zeile", [0, 17, 1234])
def test_app_rechnet_wie_der_holdout(zeile):
    """Ein Bild durch die App, verglichen mit `p_ens` aus der Holdout-Datei.

    Der Holdout wird dabei NICHT neu gerechnet. Gelesen wird die CSV, die am
    09.08.2026 einmal entstanden ist; die Sperrdatei daneben bleibt
    unangetastet.

    Warum drei Zeilen und nicht eine: eine einzige koennte zufaellig auch dann
    passen, wenn die Vorverarbeitung leicht abweicht, weil das Bild in dem
    Bereich flach ist. Drei ueber die Datei verteilte Bilder sind billig, der
    Test rechnet ohnehin nur Vorwaertsschritte.
    """
    pd = pytest.importorskip("pandas")
    if not HOLDOUT_CSV.is_file():
        pytest.skip(f"{HOLDOUT_CSV} fehlt")
    if not BILDER.is_dir():
        pytest.skip(f"{BILDER} fehlt")
    if not KALIBRIERUNG.is_file():
        pytest.skip(f"{KALIBRIERUNG} fehlt")

    kal = json.loads(KALIBRIERUNG.read_text(encoding="utf-8"))
    fehlend = [c for c in kal["checkpoints"] if not (ROOT / c).is_file()]
    if fehlend:
        pytest.skip(f"Gewichte fehlen: {fehlend[0]}")

    df = pd.read_csv(HOLDOUT_CSV)
    if zeile >= len(df):
        pytest.skip(f"die Holdout-Datei hat nur {len(df)} Zeilen")
    pid = str(df.loc[zeile, "patientId"])
    erwartet = float(df.loc[zeile, "p_ens"])

    bild = BILDER / f"{pid}.png"
    if not bild.is_file():
        pytest.skip(f"{bild} fehlt")

    from PIL import Image

    # main.py laedt beim Import die fuenf Gewichte. Das dauert einige Sekunden
    # und passiert deshalb genau einmal, nicht je Parameter.
    main = _app()

    with Image.open(bild) as im:
        # Die App bekommt im Betrieb ein RGB-Bild aus dem Upload, der
        # Trainingslader ein L-Bild aus der Platte. Genau diese Stelle war der
        # Unterschied, den der Umbau beseitigt hat: `model_input` wandelt
        # zuerst nach Graustufen und skaliert danach, so wie RsnaDataset.
        # Deshalb wird hier ABSICHTLICH ueber RGB gegangen; ginge es nur
        # zufaellig gut, weil das Testbild schon grau ist, wuerde der Test die
        # Falle nicht stellen.
        p_views, feld, p_folds, _ = main.ensemble_scores(im.convert("RGB"))

    gerechnet = float(p_views[0])
    assert abs(gerechnet - erwartet) < TOLERANZ, (
        f"{pid}: App {gerechnet:.8f} gegen Holdout {erwartet:.8f}, "
        f"Abweichung {abs(gerechnet - erwartet):.2e}. Die App liefert eine "
        f"andere Zahl als die Auswertung, die im Ergebnistext steht."
    )

    # Die Einzelwerte gleich mit, sie stehen in derselben Datei. Faellt nur
    # EIN Fold auf, ist es ein vertauschtes Gewicht oder eine vertauschte
    # Kurve; faellt alles auf, ist es die Vorverarbeitung.
    for k, p in enumerate(p_folds):
        soll = float(df.loc[zeile, f"p_kal_f{k}"])
        assert abs(p - soll) < TOLERANZ, (
            f"{pid}, Fold {k}: App {p:.8f} gegen Holdout {soll:.8f}")

    assert feld.shape == (14, 14)
    assert 0.0 <= float(feld.min()) and float(feld.max()) <= 1.0


_APP = None


def _app():
    """main.py einmal importieren. Der Import laedt die fuenf Gewichte."""
    global _APP
    if _APP is None:
        if str(SERVING) not in sys.path:
            sys.path.insert(0, str(SERVING))
        import os
        # main.py bricht bei gesetztem THRESHOLD absichtlich ab. In einer Shell,
        # in der die Variable aus alten Zeiten noch steht, soll der Test die
        # App pruefen und nicht die Shell.
        os.environ.pop("THRESHOLD", None)
        # Dasselbe fuer TEST_DIR: es zeigte bis zum 09.08.2026 auf den
        # Kermany-Testordner und bricht seitdem ebenfalls absichtlich ab.
        os.environ.pop("TEST_DIR", None)
        _APP = _lade("serving_main", SERVING / "main.py")
    return _APP


def test_platt_gleich_wie_im_holdout():
    """Dieselbe Formel und dasselbe eps wie `rsna_holdout.platt_apply`."""
    if not KALIBRIERUNG.is_file():
        pytest.skip(f"{KALIBRIERUNG} fehlt")
    # rsna_holdout.py laesst sich nicht importieren, ohne rsna_train mitzuziehen
    # und `_repo_path` zu setzen. Deshalb steht die Formel hier woertlich noch
    # einmal, abgeschrieben aus `rsna_holdout.platt_apply`. Weicht main.py
    # davon ab, faellt der Test, und genau das ist sein Zweck. Der Vergleich
    # laeuft mit atol=0 und rtol=0: es geht nicht um "ungefaehr dieselbe
    # Kurve", sondern um dieselbe Rechnung.
    def logit(p, eps=1e-6):
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def platt(p, a, b):
        return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))

    main = _app()
    p = np.array([0.0, 1e-9, 0.001, 0.2003, 0.5, 0.9999, 1.0])
    for a, b in ((0.5809428850556021, -1.0344684452790784),
                 (0.8283144233238171, -1.4498154833747867)):
        assert np.allclose(main.platt_apply(p, a, b), platt(p, a, b), atol=0, rtol=0)


# --------------------------------------------------------------------------
# Die drei Zusaetze vom 09.08.2026
# --------------------------------------------------------------------------
# Sie pruefen nicht das Modell, sondern drei Entscheidungen, die still brechen
# koennen: eine Auswahl, die nicht mehr zu ihrer Quelle passt, eine
# Referenzverteilung, die nicht mehr sortiert ist, und eine Karte, die anfaengt,
# jedes Bild auf sein eigenes Maximum zu strecken.

MANIFEST = SERVING / "samples" / "manifest.json"
REFERENZ = SERVING / "model" / "referenz_dev.json"


def test_demobilder_passen_zu_ihrer_quelle():
    """Jedes Demobild liegt da, ist eindeutig, und stammt aus dem Holdout.

    Der Punkt ist die letzte Pruefung: das Manifest nennt zu jedem Bild die
    Wahrscheinlichkeit, mit der es ausgewaehlt wurde. Weicht sie von der Zeile
    in `holdout.csv` ab, ist das Manifest aus einer anderen Quelle geschrieben
    worden als der, die im Kopf der Datei steht, und die Auswahlregel waere
    nicht mehr nachvollziehbar.
    """
    if not MANIFEST.is_file():
        pytest.skip(f"{MANIFEST} fehlt")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bilder = manifest["bilder"]

    ids = [b["id"] for b in bilder]
    assert len(ids) == len(set(ids)), f"doppelte Kennung: {ids}"
    for b in bilder:
        assert (MANIFEST.parent / b["file"]).is_file(), f"{b['file']} fehlt"
        # Klasse und Trainingslabel muessen zueinander passen, sonst zeigt die
        # Oberflaeche eine Wahrheit an, die nicht die gemessene ist.
        assert (b["rsna_class"] == "Lung Opacity") == (b["y"] == 1), b["id"]

    zellen = {(b["category"], b["viewpos"]) for b in bilder}
    assert len(zellen) == len(bilder) == 6, f"erwartet 6 Zellen, sind {zellen}"

    if not HOLDOUT_CSV.is_file():
        pytest.skip(f"{HOLDOUT_CSV} fehlt")
    import csv

    with open(HOLDOUT_CSV, encoding="utf-8") as fh:
        p_ens = {z["patientId"]: float(z["p_ens"]) for z in csv.DictReader(fh)}
    for b in bilder:
        assert b["patientId"] in p_ens, f"{b['id']} steht nicht im Holdout"
        assert abs(p_ens[b["patientId"]] - b["p_ens_holdout"]) < 5e-5, (
            f"{b['id']}: Manifest sagt {b['p_ens_holdout']}, holdout.csv sagt "
            f"{p_ens[b['patientId']]:.4f}")


def test_referenzverteilung_ist_brauchbar():
    """Sortiert, vollstaendig, und die Einordnung waechst monoton."""
    if not REFERENZ.is_file():
        pytest.skip(f"{REFERENZ} fehlt")
    ref = json.loads(REFERENZ.read_text(encoding="utf-8"))
    raster = ref["raster"]
    assert ref["n"] == 22872, ref["n"]
    assert raster == sorted(raster), "das Raster ist nicht sortiert"
    assert 0.0 <= raster[0] and raster[-1] <= 1.0

    main = _app()
    # Monoton, und die Schwelle muss im mittleren Bereich landen. Laege sie bei
    # 1 oder bei 99 Prozent, waere die Verteilung nicht die des ausgelieferten
    # Arms, und der Satz in der Oberflaeche waere irrefuehrend statt falsch -
    # also genau die Sorte Fehler, die niemand bemerkt.
    werte = [main.reference_percentile(x) for x in (0.0, 0.05, 0.2003, 0.5, 0.9, 1.0)]
    assert werte == sorted(werte), werte
    assert 40.0 < main.reference_percentile(float(main.THRESHOLD)) < 80.0


def test_kopffeld_ebene_wird_nicht_je_bild_gestreckt():
    """Ein Feld ohne Ausschlag muss unsichtbar sein.

    Das ist die eine Eigenschaft der Karte, die man beim Aufhuebschen verliert,
    ohne es zu merken: streckt man jedes Bild auf sein eigenes Maximum, sieht
    ein Feld, das nichts sagt, genauso deutlich aus wie ein starkes. Der Pegel
    des Kopfes ist unkalibriert (62 Prozent Alarm auf gesunden Bildern, Phase
    5b), also ist genau diese Streckung die Aussage, die die Messung nicht
    traegt.
    """
    stages = _lade("serving_stages", SERVING / "stages.py")
    from PIL import Image
    import base64
    import io

    def alpha(feld):
        roh = base64.b64decode(stages.render_head_field_layer(feld))
        bild = Image.open(io.BytesIO(roh))
        assert bild.mode == "RGBA", bild.mode
        return np.asarray(bild)[..., 3]

    leer = alpha(np.zeros((14, 14), dtype=np.float32))
    assert leer.max() == 0, f"ein leeres Feld ist sichtbar (Alpha bis {leer.max()})"

    schwach = alpha(np.full((14, 14), 0.02, dtype=np.float32))
    assert schwach.max() <= 5, f"ein schwaches Feld leuchtet (Alpha {schwach.max()})"

    voll = alpha(np.ones((14, 14), dtype=np.float32))
    assert voll.min() == voll.max() == int(0.75 * 255), voll.max()

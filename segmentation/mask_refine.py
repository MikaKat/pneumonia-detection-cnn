"""Masken-Verfeinerung gegen den klassenkorrelierten Formen-Leak (lung_area).

Motivation: Die Diagnostik hat gezeigt, dass die segmentierte Lungenfläche die
Klasse verrät (lung_area AUC ~0.255) - Pneumonie-Lungen werden untersegmentiert,
also kleiner. Diese zwei Schritte holen fehlende Fläche geometrisch zurück und
machen die Maskenfläche klassenunabhängiger:

  * KONVEXE HÜLLE je Lunge: füllt löchrige/fleckige Untersegmentierung (der
    Konsolidierungs-Fall). Nebenwirkung: kann die Herztaille links leicht
    überfüllen - deshalb gegen lung_area messen und ggf. abschalten.
  * SYMMETRIE-FÜLLUNG: fehlt eine ganze Seite stark, wird die größere Lunge an
    der Mediastinal-Achse (Mittel der beiden Schwerpunkte) gespiegelt und der
    kleinen Seite hinzugefügt - nur auf der kleinen Seite, gegen Herz-Übergriff.

Beide Schritte sind über die Flags unten einzeln schaltbar, um ihren Effekt auf
die lung_area-AUC getrennt zu prüfen (siehe segmentation/eval_leak.py).
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

USE_CONVEX_HULL = False
USE_SYMMETRY = True
KEEP_FRAC = 0.05        # Komponenten >= 5 % der größten behalten (2. Lunge nicht verlieren)
RATIO_THRESH = 0.60     # Symmetrie nur, wenn kleine/große Lunge < 60 % (klare Asymmetrie)


def _clean(binary):
    """Closing -> Löcher füllen -> die (bis zu) 2 größten Komponenten behalten."""
    m = ndi.binary_closing(binary, structure=np.ones((5, 5)))
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n <= 1:
        return m.astype(bool)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    largest = sizes[order[0]]
    keep = [c for c in order[:2] if sizes[c] >= KEEP_FRAC * largest]
    return np.isin(lbl, keep)


def _components(binary):
    """Bis zu 2 größte Komponenten als Liste (fläche, maske, schwerpunkt_xy), abst."""
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    comps = [(stats[i, cv2.CC_STAT_AREA], lbl == i, cent[i]) for i in range(1, n)]
    comps.sort(key=lambda c: -c[0])
    return comps[:2]


def _convex_hull_per_lung(binary):
    """Konvexe Hülle jeder der 2 Lungen einzeln (nicht der Vereinigung - sonst
    würde die Brücke über das Mediastinum das Herz einschließen)."""
    out = binary.astype(bool).copy()
    for _, m, _ in _components(binary):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            tmp = np.zeros(binary.shape, np.uint8)
            cv2.fillConvexPoly(tmp, cv2.convexHull(cnt), 1)
            out |= tmp.astype(bool)
    return out


def _symmetry_fill(binary):
    """Bei starker Asymmetrie die größere Lunge an der Mediastinal-Achse spiegeln
    und der kleinen Seite hinzufügen. Nur auf der kleinen Seite (kein Herz-Übergriff)."""
    comps = _components(binary)
    if len(comps) < 2:
        return binary.astype(bool)                 # nur eine Lunge gefunden -> nicht raten
    (a_big, m_big, c_big), (a_small, _, c_small) = comps
    if a_small / max(a_big, 1) > RATIO_THRESH:
        return binary.astype(bool)                 # symmetrisch genug -> nichts tun

    H, W = binary.shape
    axis = (c_big[0] + c_small[0]) / 2.0           # Mediastinum ~ Mittel der Schwerpunkte
    ys, xs = np.where(m_big)
    xr = np.round(2 * axis - xs).astype(int)       # an der Achse spiegeln
    valid = (xr >= 0) & (xr < W)
    mirror = np.zeros(binary.shape, bool)
    mirror[ys[valid], xr[valid]] = True
    # nur auf der Seite der kleinen Lunge behalten:
    if c_small[0] < axis:
        mirror[:, int(axis):] = False
    else:
        mirror[:, :int(axis)] = False
    return binary.astype(bool) | mirror


def refine_mask(binary):
    """Vollständige Verfeinerung: säubern -> (konvexe Hülle) -> (Symmetrie)."""
    m = _clean(binary)
    if USE_CONVEX_HULL:
        m = _convex_hull_per_lung(m)
    if USE_SYMMETRY:
        m = _symmetry_fill(m)
    return m

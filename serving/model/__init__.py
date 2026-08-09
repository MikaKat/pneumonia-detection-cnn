"""Modell-Paket. Re-exportiert die Bauhelfer, damit sowohl
`from model.model import build_model` als auch `from model import build_model`
funktionieren."""

from model.model import (HEAD_GRID, ClassifierView, TwoHeadNet, build_model,
                         build_two_head_model)

__all__ = ["build_model", "build_two_head_model", "TwoHeadNet",
           "ClassifierView", "HEAD_GRID"]

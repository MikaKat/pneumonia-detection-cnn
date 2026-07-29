"""Modell-Paket. Re-exportiert build_model, damit sowohl
`from model.model import build_model` als auch `from model import build_model`
funktionieren."""

from model.model import build_model

__all__ = ["build_model"]

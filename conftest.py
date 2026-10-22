"""Konfiguracja pytest: katalog projektu na ścieżce importów."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

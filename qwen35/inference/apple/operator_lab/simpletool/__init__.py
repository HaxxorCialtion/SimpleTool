"""Independent SimpleTool batched fork decoder."""
import sys
from pathlib import Path

# Optional dependencies stay local; do not modify the other agent's venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))

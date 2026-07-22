import sys
from pathlib import Path

# Ensure project root is on sys.path so all packages (experiments, model, etc.)
# are importable regardless of pytest invocation mode.
sys.path.insert(0, str(Path(__file__).parent))

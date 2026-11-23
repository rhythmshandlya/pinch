"""Electrical power budget from simulation: a short name for sim/circuit_check.py.

The robot (PARAMS.model_file) runs in the twin with the circuit model of its wiring (crab/wiring.py Circuit,
configs/wiring_v2.toml): standing, walking forward and sideways, a get-up swing, then the battery minutes of a mixed day.

    python sim/power.py [--bam] [--seconds 8]      # the same as python sim/circuit_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim import circuit_check  # noqa: E402

if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if a != "--v2"]       # (the old switch: there is one robot now)
    sys.exit(circuit_check.main())

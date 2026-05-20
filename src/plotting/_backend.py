"""Force the non-interactive Agg backend before any ``pyplot`` import.

Every Stage 5 plotting module imports this module first so that a fresh
``import matplotlib.pyplot`` later in the same process cannot pick up an
interactive backend (e.g. when running headless on the grader).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

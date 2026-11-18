"""Location of the reconstruction output used by the analysis scripts.

Set ``RECON_DATA_DIR`` in the environment or in a ``.env`` file at the repository
root (see ``.env.example``). The directory is expected to contain one
``<phantom>-<dataset>/<dose>mGy/<n_proj>proj/recon/`` tree per reconstruction
setup, as written by ``recon/`` (``OUTPUT_DIR`` in ``recon/config.py``).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RECON_DATA_DIR = Path(os.environ.get("RECON_DATA_DIR", "~/Scratch/data/recon")).expanduser()

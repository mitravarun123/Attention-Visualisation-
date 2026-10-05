"""Start the built React interface with the Python model API."""
import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

if __name__ == '__main__':
    if not (ROOT / 'frontend/dist/index.html').exists():
        raise SystemExit('Build the frontend first: cd frontend && npm install && npm run build')
    uvicorn.run('app.api:app', host='127.0.0.1', port=8501)

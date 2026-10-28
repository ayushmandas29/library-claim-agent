# Library Contents Claim Agent

Status: scaffold. Schema + totals-in-code are done and tested; pipeline stages are next.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY
PYTHONPATH=src pytest -q
python scripts/make_marker.py --size-cm 15 --id 7 --out marker.png
```

## Rules baked into code
- Unknown = `None`/empty. Never a guessed value.
- A price must carry source + retrieval date, or the packet fails validation.
- Totals come from `totals.py`, never from a model.

## Reference app: kept / changed / thrown away
TODO after running it.

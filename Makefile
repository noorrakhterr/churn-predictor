.PHONY: setup data preprocess train explain app test format lint all

# ── Environment ───────────────────────────────────────────────────────────────
setup:
	pip install --upgrade pip
	pip install -r requirements.txt

# ── Pipeline steps ────────────────────────────────────────────────────────────
data:
	python -m src.generate_data

preprocess:
	python -m src.preprocess

train:
	python -m src.train

explain:
	python -m src.explain

# ── App ───────────────────────────────────────────────────────────────────────
app:
	streamlit run app/app.py

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

format:
	black src/ app/ tests/
	ruff check --fix src/ app/ tests/

lint:
	ruff check src/ app/ tests/

# ── Full pipeline (data → model → explanations) ───────────────────────────────
all: data preprocess train explain

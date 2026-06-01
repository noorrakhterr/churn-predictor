.PHONY: setup train app test format

# Install all dependencies into the active virtual environment
setup:
	pip install --upgrade pip
	pip install -r requirements.txt

# Generate synthetic data, then train and evaluate the model
train:
	python src/generate_data.py
	python src/preprocess.py
	python src/train.py

# Launch the Streamlit dashboard
app:
	streamlit run app/app.py

# Run the test suite
test:
	pytest tests/ -v

# Auto-format and lint the codebase
format:
	black src/ app/ tests/
	ruff check src/ app/ tests/ --fix

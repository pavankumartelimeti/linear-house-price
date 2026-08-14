.PHONY: install data train test lint api app docker-build docker-run clean

install:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset

train: data
	python -m src.models.train

test:
	pytest tests/ -v

lint:
	ruff check src/ api/ tests/ app/

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

notebook:
	jupyter notebook notebooks/01_full_analysis.ipynb

docker-build:
	docker build -t house-price-regression .

docker-run:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

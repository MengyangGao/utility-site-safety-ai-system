ENV_NAME ?= utility-safety-ai
CONDA ?= conda
RUN := $(CONDA) run -n $(ENV_NAME)
PYTHON := $(RUN) python
CLI := $(RUN) utility-safety-ai

OUTPUT ?= outputs/demo
RUN_ID ?= portfolio-demo
MODEL ?= models/ppe_yolo11n.pt
GENERAL_MODEL ?= models/yolo11n.pt
DATA ?= construction-ppe.yaml

.PHONY: help setup env install fetch-model test test-coverage lint typecheck web-check verify web \
	demo demo-ppe demo-video demo-camera show-latest train-ppe validate benchmark export-model \
	assets clean

help:
	@echo "setup          Create the Python 3.11 Conda environment and install the package"
	@echo "install        Refresh editable package/dev dependencies in an existing environment"
	@echo "fetch-model    Download yolo11n.pt into models/ with SHA-256 metadata"
	@echo "verify         Run tests with coverage, Ruff, mypy, and Web syntax check"
	@echo "demo           Clean-clone person + zone image demo"
	@echo "demo-ppe       PPE image demo; requires MODEL (default models/ppe_yolo11n.pt)"
	@echo "demo-video     Clean-clone person + zone video demo"
	@echo "demo-camera    Ten-second webcam demo"
	@echo "show-latest    Print the last successful run below OUTPUT"
	@echo "web            Start the Streamlit app"

setup: env install

env:
	$(CONDA) env create -f environment.yml

install:
	$(RUN) pip install -e ".[dev]"

fetch-model:
	$(CLI) fetch-model --model yolo11n.pt --output models

test:
	$(PYTHON) -m pytest -q

test-coverage:
	$(PYTHON) -m pytest -q --cov=utility_safety_ai --cov-report=term-missing --cov-fail-under=70

lint:
	$(RUN) ruff check .

typecheck:
	$(RUN) mypy src/utility_safety_ai

web-check:
	$(PYTHON) -m py_compile app.py

verify: test-coverage lint typecheck web-check

web:
	$(RUN) streamlit run app.py

demo:
	$(CLI) infer-image \
	  --source examples/sample_images/construction_zone_01.jpg \
	  --model $(GENERAL_MODEL) \
	  --zones examples/zones_construction_zone_01.yaml \
	  --output $(OUTPUT) \
	  --run-id $(RUN_ID) \
	  --overwrite \
	  --blur-faces

demo-ppe:
	$(CLI) infer-image \
	  --source examples/sample_images/construction_site_ppe_01.jpg \
	  --model $(MODEL) \
	  --zones examples/zones_construction_site_ppe_01.yaml \
	  --output $(OUTPUT) \
	  --run-id $(RUN_ID)-ppe \
	  --overwrite \
	  --blur-faces

demo-video:
	$(CLI) infer-video \
	  --source examples/sample_videos/construction_rebar_pexels_10294768.mp4 \
	  --model $(GENERAL_MODEL) \
	  --zones examples/zones_construction_rebar_pexels_10294768.yaml \
	  --output $(OUTPUT) \
	  --run-id $(RUN_ID)-video \
	  --overwrite \
	  --max-frames 120 \
	  --blur-faces

demo-camera:
	$(CLI) infer-camera \
	  --source 0 \
	  --output $(OUTPUT) \
	  --run-id $(RUN_ID)-camera \
	  --overwrite \
	  --duration 10 \
	  --blur-faces

show-latest:
	$(PYTHON) -c "from pathlib import Path; from utility_safety_ai.utils.paths import resolve_latest_run; p = resolve_latest_run(Path('$(OUTPUT)')); print(p if p else 'No completed run')"

train-ppe:
	$(CLI) train \
	  --data $(DATA) \
	  --model yolo11n.pt \
	  --epochs 30 \
	  --imgsz 640 \
	  --batch 16 \
	  --project runs/train_ppe \
	  --name ppe_yolo11n

validate:
	$(PYTHON) scripts/validate_ppe_model.py \
	  --model $(MODEL) \
	  --data $(DATA) \
	  --output outputs/validation/ppe_yolo11n

benchmark:
	$(PYTHON) scripts/benchmark.py \
	  --model $(MODEL) \
	  --output outputs/benchmark/ppe_yolo11n

export-model:
	$(CLI) export-model \
	  --model $(MODEL) \
	  --format onnx \
	  --output outputs/export

assets:
	$(PYTHON) scripts/make_demo_assets.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; shutil.rmtree(Path('outputs'), ignore_errors=True)"

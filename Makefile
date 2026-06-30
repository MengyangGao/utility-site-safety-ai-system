.PHONY: install test demo demo-all demo-video demo-video-full-ppe demo-camera train-ppe validate validate-all benchmark lint clean assets

install:
	conda env create -f environment.yml || true
	conda run -n utility-safety-ai pip install -e ".[dev]"

test:
	pytest -q

demo:
	python scripts/make_demo_assets.py
	utility-safety-ai infer-image \
	  --source examples/sample_images/construction_zone_01.jpg \
	  --zones examples/zones_construction_zone_01.yaml \
	  --output outputs/demo-ppe \
	  --blur-faces

demo-all:
	utility-safety-ai infer-image \
	  --source examples/sample_images/construction_zone_01.jpg \
	  --zones examples/zones_construction_zone_01.yaml \
	  --output outputs/demo-ppe/construction_zone_01 \
	  --blur-faces
	utility-safety-ai infer-image \
	  --source examples/sample_images/construction_site_ppe_01.jpg \
	  --zones examples/zones_construction_site_ppe_01.yaml \
	  --output outputs/demo-ppe/construction_site_ppe_01 \
	  --blur-faces
	utility-safety-ai infer-image \
	  --source examples/sample_images/solar_farm_01.jpg \
	  --zones examples/zones_solar_farm_01.yaml \
	  --output outputs/demo-ppe/solar_farm_01 \
	  --blur-faces
	utility-safety-ai infer-image \
	  --source examples/sample_images/construction_worker_gloves_01.jpg \
	  --output outputs/demo-ppe/gloves_demo \
	  --blur-faces

demo-video:
	utility-safety-ai infer-video \
	  --source examples/sample_videos/construction_site_pan.mp4 \
	  --zones examples/zones_construction_zone_01.yaml \
	  --output outputs/demo-ppe-video \
	  --blur-faces

demo-video-full-ppe:
	utility-safety-ai infer-video \
	  --source examples/sample_videos/construction_ppe_pan.mp4 \
	  --output outputs/demo-ppe-full-ppe-video \
	  --blur-faces

demo-camera:
	utility-safety-ai infer-camera \
	  --source 0 \
	  --zones examples/zones_construction_zone_01.yaml \
	  --output outputs/demo-camera \
	  --blur-faces \
	  --duration 10

train-ppe:
	utility-safety-ai train \
	  --data construction-ppe.yaml \
	  --model yolo11n.pt \
	  --epochs 30 \
	  --imgsz 640 \
	  --project runs/train_ppe \
	  --name ppe_yolo11n_30ep

validate:
	python scripts/validate_ppe_model.py --model models/ppe_yolo11n.pt --output outputs/validation/yolo11n

validate-all:
	python scripts/validate_ppe_model.py --model models/ppe_yolo11n.pt --output outputs/validation/yolo11n
	python scripts/validate_ppe_model.py --model models/ppe_yolo11s.pt --output outputs/validation/yolo11s

benchmark:
	python scripts/benchmark.py --model models/ppe_yolo11n.pt --output outputs/benchmark

lint:
	ruff check .

clean:
	rm -rf outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

assets:
	python scripts/make_demo_assets.py

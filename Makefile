.PHONY: augment_old
augment_old:
	python -m src.generate_old_data \
		--data-dir data/2025_data \
		--output-dir data/processed/old_augmented \
		--samples-per-positives 3 \
		--min-history 1000 \
		--max-history 5000 \
		--min-online 10 \
		--max-online 1000 \
		--seed 42

.PHONY: split
split:
	python -m src.create_folds \
		--x-path data/2026_data/X_train.parquet \
		--y-index-path data/2026_data/y_train_index.parquet \
		--output-path data/processed/folds.parquet \
		--n-splits 5 \
		--seed 42

MIN_POST_BREAK ?= 5
NUM_WORKERS ?= 12
SERIES_PER_PART ?= 50


.PHONY: train_features_ema5 train_features_ema10
.PHONY: augment_features_ema5 augment_features_ema10
.PHONY: test_features_ema5 test_features_ema10

features:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	python -m src.generate_features \
		--x-path $(X_PATH) \
		--y-path $(Y_PATH) \
		--y-index-path $(Y_INDEX_PATH) \
		--output-path $(OUTPUT_PATH) \
		--num-workers $(NUM_WORKERS) \
		--series-per-chunk $(SERIES_PER_PART)

train_features_v3:
	$(MAKE) features \
		X_PATH=data/2026_data/X_train.parquet \
		Y_PATH=data/2026_data/y_train.parquet \
		Y_INDEX_PATH=data/2026_data/y_train_index.parquet \
		OUTPUT_PATH=data/final/$@.parquet

augment_features_v3:
	$(MAKE) features \
		X_PATH=data/processed/old_augmented/X_train.parquet \
		Y_PATH=data/processed/old_augmented/y_train.parquet \
		Y_INDEX_PATH=data/processed/old_augmented/y_index.parquet \
		OUTPUT_PATH=data/final/$@.parquet

test_features_v3:
	$(MAKE) features \
		X_PATH=data/2026_data/X_test.reduced.parquet \
		Y_PATH=data/2026_data/y_test.reduced.parquet \
		Y_INDEX_PATH=data/2026_data/y_test_index.reduced.parquet \
		OUTPUT_PATH=data/final/$@.parquet
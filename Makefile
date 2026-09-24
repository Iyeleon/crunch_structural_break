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
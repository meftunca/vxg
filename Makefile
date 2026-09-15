.PHONY: test model bench shaders format

test:
	bash scripts/check.sh

model:
	bash scripts/check_model.sh

bench:
	bash scripts/bench.sh

shaders:
	python3 scripts/generate_shaders.py --check

format:
	cargo fmt --all

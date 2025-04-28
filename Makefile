# Make PROJ_DIR the parent of this file
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

TORCH_ENV := $(PROJ_DIR)/.venv
KERAS_ENV := $(PROJ_DIR)/.venv-keras

# pytorch Environment
# ---

.PHONY: install-uv
install-uv:
	# Install the uv command line tool if not already installed
	@echo ""
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "🔧 Installing uv..."; \
		pip install --upgrade pip; \
		pip install --upgrade uv; \
	else \
		echo "✅ uv is already installed."; \
	fi


.PHONY: torch-env
torch-env:
	# Create a virtual environment using pytorch dependencies




.PHONY: torch-compile
torch-compile:
	 uv export --group pytorch --format requirements-txt -o requirements-torch.txt


# Tensorflow Environment
# ---
.PHONY: keras-env
keras-env:
	# Create a virtual environment using tensorflow dependencies
	@echo ""
	...


.PHONY: keras-compile
keras-compile:
	uv export --group keras --format requirements-txt -o requirements-keras.txt

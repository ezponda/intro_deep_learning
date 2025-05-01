# Make PROJ_DIR the parent of this file
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

TORCH_ENV := $(PROJ_DIR)/.venv-torch
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


.PHONY: torch-env-craete
torch-env-create:
	# Create a virtual environment using pytorch dependencies if it doesn't exist
	@if [ ! -d "$(TORCH_ENV)" ]; then \
  		echo "🔧 Creating virtual environment..." uv init $(TORCH_ENV); else \
		echo "✅ Virtual environment already exists."; fi


.PHONY: torch-setup
torch-setup: install-uv torch-env-create
	# Install the initial requirements for the pytorch environment
	@echo "Installing initial requirements..."
	@uv install --group pytorch --format requirements-txt \
 		-o $(TORCH_ENV)/requirements/requirements-torch.txt

.PHONY: torch-compile
torch-compile:
	@echo "Compiling requirements for torch..."
	@uv export --group pytorch --format requirements-txt \
 		-o requirements/requirements-torch.txt


# Tensorflow Environment
# ---
.PHONY: keras-env
keras-env:
	# Create a virtual environment using tensorflow dependencies if it doesn't exist
	...


.PHONY: keras-compile
keras-compile:
	...

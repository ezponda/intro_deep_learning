# Make PROJ_DIR the parent of this file
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

VENV := $(PROJ_DIR)/.venv
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
torch-compile: keras-activate
	@echo "Compiling requirements for torch..."
	@uv export --group pytorch --format requirements-txt \
 		-o requirements/requirements-torch.txt


# Tensorflow Environment
# ---
.PHONY: keras-env-craete
keras-env-create:
	# Create a virtual environment using tensorflow dependencies if it doesn't exist
	@if [ ! -d "$(KERAS_ENV)" ]; then \
  		echo "🔧 Creating virtual environment..." uv init $(KERAS_ENV); else \
		echo "✅ Virtual environment already exists."; fi

.PHONY: keras-setup
keras-setup: install-uv keras-env-create
	# Install the initial requirements for the keras environment
	@echo "Installing initial requirements..."
	@uv install --format requirements-txt \
 		-o $(KERAS_ENV)/requirements/requirements-keras.txt

.PHONY: keras-activate
keras-activate:
	# Activate the keras environment if not already activated
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "🔧 Activating keras environment..."; \
		. $(KERAS_ENV)/bin/activate; \
	else \
		echo "✅ Keras environment already activated."; \
	fi

.PHONY: keras-compile
keras-compile: keras-activate
	@echo "Compiling requirements for keras..."
	@uv export --format requirements-txt \
 		-o requirements/requirements-keras.txt

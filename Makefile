# Make PROJ_DIR the parent of this file
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# pytorch Environment
# ---

.PHONY: torch-env
torch-env:
	# Create a virtual environment using pytorch dependencies
	@echo ""
	...


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

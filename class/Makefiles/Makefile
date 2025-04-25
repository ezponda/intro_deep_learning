# Make PROJ_DIR the parent of this file
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

NEW_VENV := $(PROJ_DIR)/.venv

.PHONY: setup add-deps update-env

# Auxiliary targets
# ---

clean:
	@echo "Cleaning up..."
	rm -rf $(NEW_VENV)
	rm -f final-requirements.txt

create:
	@echo "Creating virtual environment..."
	@if [ ! -d $(NEW_VENV)]; then \
  		echo "🔧 Creating virtual environment..."; python3 -m venv $(NEW_VENV); else \
		echo "✅ Virtual environment already exists."; fi

activate: create
	@echo "Activating virtual environment..."
	@. .venv/bin/activate

# Main targets
# ----

# Step 1: Create environment and install initial dependencies
setup: clean activate activate
	@echo "Installing initial requirements..."
	pip install -r initial-requirements.txt

# Step 2: Add new dependency by installing into the environment
add-deps: activate
	pip install matplotlib

# Step 3: Freeze current env state into final-requirements.txt
freeze-env: activate
	pip freeze > final-requirements.txt

# Step 4: Update env using final-requirements.txt
update-env:
	. .venv/bin/activate && pip install -r final-requirements.txt

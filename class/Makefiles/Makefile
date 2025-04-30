PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NEW_VENV := $(PROJ_DIR)/.venv

.PHONY: setup add-deps update-env

clean:
	@echo "Cleaning up..."
	rm -rf $(NEW_VENV)
	rm -f final-requirements.txt

create:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(NEW_VENV)" ]; then \
  		echo "🔧 Creating virtual environment..."; python3 -m venv $(NEW_VENV); else \
		echo "✅ Virtual environment already exists."; fi

setup: clean create
	@echo "Installing initial requirements..."
	@$(NEW_VENV)/bin/pip install -r initial-requirements.txt

add-deps: create
	@$(NEW_VENV)/bin/pip install matplotlib

freeze-env: create
	@$(NEW_VENV)/bin/pip freeze > final-requirements.txt

update-env: create
	@$(NEW_VENV)/bin/pip install -r final-requirements.txt

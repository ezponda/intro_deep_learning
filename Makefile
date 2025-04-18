#

.PHONY: torch-env
torch-env:
	# Create a virtual environment using pytorch dependencies
	@echo ""
	uv


.PHONY: torch-compile
torch-compile:
	 uv export --group pytorch --format requirements-txt -o requirements-torch.txt
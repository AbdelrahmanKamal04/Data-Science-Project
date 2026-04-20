#################################################################################
# CONFIGURATION                                                                 #
#################################################################################

PROJECT_NAME        = Data-Science-Project
PYTHON_VERSION      = 3.10
PYTHON_INTERPRETER  = poetry run python
MODULE              = Data_Science_Module
export PYTHONPATH  := .

REPORTS_DIR         = reports
VALIDATION_REPORT   = $(REPORTS_DIR)/validation_report.json

#################################################################################
# ENVIRONMENT SETUP                                                             #
#################################################################################

## One-command first-time setup (create env + install all deps)
.PHONY: setup
setup: create_environment requirements
	@echo ""
	@echo ">>> Setup complete. Run 'make help' to see available commands."

## Create the Poetry virtual environment
.PHONY: create_environment
create_environment:
	@echo ">>> Creating Poetry environment (Python $(PYTHON_VERSION))..."
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Activate with: poetry shell"

## Install all dependencies (runtime + dev) from pyproject.toml
.PHONY: requirements
requirements:
	poetry install --with dev
	@echo ">>> Dependencies installed."

## Install runtime dependencies only (for CI / production)
.PHONY: requirements-prod
requirements-prod:
	poetry install --without dev

## Install using pip (alternative to Poetry)
.PHONY: pip-install
pip-install:
	python -m pip install -r requirements.txt
	@echo ">>> pip install complete."

#################################################################################
# DATA VALIDATION                                                               #
#################################################################################

## Run all 8-dimension data validation checks and save JSON report
.PHONY: validate
validate:
	@echo ">>> Running data validation..."
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON_INTERPRETER) -m $(MODULE).validate
	@echo ">>> Validation report written to $(VALIDATION_REPORT)"

#################################################################################
# DATA PIPELINE                                                                 #
#################################################################################

## Full pipeline: merge raw data → validate → features → train
.PHONY: all
all: data features train
	@echo ">>> Full pipeline complete."

## Merge raw transaction + identity files, then validate
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m $(MODULE).dataset

## Engineer features (writes to data/interim and data/processed)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m $(MODULE).features

## Train model (writes artefacts to models/)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m $(MODULE).modeling.train

## Run predictions with the trained model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m $(MODULE).modeling.predict

#################################################################################
# CODE QUALITY                                                                  #
#################################################################################

## Lint (check only, no writes)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Auto-fix formatting and lint issues
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run test suite with coverage report
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ --cov=$(MODULE) --cov-report=term-missing

#################################################################################
# HOUSEKEEPING                                                                  #
#################################################################################

## Remove compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name "*.egg-info" -delete
	@echo ">>> Clean complete."

## Remove generated reports (keeps directory structure)
.PHONY: clean-reports
clean-reports:
	find $(REPORTS_DIR) -name "*.json" -delete
	find $(REPORTS_DIR)/figures -type f -delete
	@echo ">>> Reports cleared."

## Remove trained model artefacts
.PHONY: clean-models
clean-models:
	find models/ -type f ! -name ".gitkeep" -delete
	@echo ">>> Models cleared."

#################################################################################
# SELF-DOCUMENTING HELP                                                         #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('Available rules:\n')
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# Makefile for setting up the Python environment

# Variables
VENV_DIR := .venv
PYTHON := python
POETRY := poetry
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := $(VENV_DIR)/bin/activate

# Targets
.PHONY: install

install: $(VENV_DIR) $(ACTIVATE) pyproject.toml
	@echo "Installing Python dependencies..."
	. $(ACTIVATE) && \
	$(POETRY) install

$(VENV_DIR):
	@echo "Setting up Python environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel poetry

$(ACTIVATE): $(VENV_DIR)

clean:
	@echo "Cleaning up Python environment..."
	@rm -rf $(VENV_DIR)
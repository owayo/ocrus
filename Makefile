.PHONY: build release install clean test fmt check help smoke wheel pytest

# Default target
.DEFAULT_GOAL := help

# Variables
BINARY_NAME := ocrus
INSTALL_PATH := /usr/local/bin

## Build Commands

build: ## Build debug version
	cargo build

release: ## Build release version
	cargo build --release

## Installation

install: release ## Build release and install to /usr/local/bin
	cp target/release/$(BINARY_NAME) $(INSTALL_PATH)/

## Development

test: ## Run tests
	cargo test

fmt: ## Format code
	cargo fmt

check: ## Run clippy and check
	cargo clippy -- -D warnings
	cargo check

clean: ## Clean build artifacts
	cargo clean

bench: ## Run benchmarks
	cargo bench

## Python bindings

smoke: ## Run the end-to-end OCR smoke check (needs the model, ~20s)
	cargo test -p ocrus-engine --release --test smoke -- --nocapture

wheel: ## Build the Python wheel into target/wheels/
	cd python && maturin build --release --out ../target/wheels

pytest: ## Run the Python binding tests (install the wheel first)
	python -m pytest python/tests -q

## Help

help: ## Show this help message
	@echo "$(BINARY_NAME) Build Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Release:"
	@echo "  Use GitHub Actions > Release > Run workflow"

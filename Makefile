# Development tasks for ocrus. Run `make` with no arguments to list the targets.
#
# Tool versions are pinned in mise.toml. When mise is available, every tool runs through
# `mise exec --`, so the pinned versions are used even when mise is not activated in the shell
# (for example when make is started from an IDE or a GUI). SYSTEM_TOOLS=1 uses the tools on PATH
# instead (the versions are then not guaranteed).
#
# Only GNU Make 3.81 features are used (the make that ships with macOS):
# no .ONESHELL, .SHELLFLAGS, $(file ...) or !=.
#

.DEFAULT_GOAL := help

BINARY_NAME := ocrus
INSTALL_PATH ?= /usr/local/bin
# Cargo.lock is committed, so resolve dependencies exactly as CI does
CARGO_FLAGS ?= --locked
PYTHON ?= python3
# Set the library path inside Python: shells and shims may strip dyld variables.
MACOS_CARGO_TEST := import os, subprocess, sys, sysconfig; env = os.environ.copy(); env["DYLD_FALLBACK_LIBRARY_PATH"] = sysconfig.get_config_var("LIBDIR") + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib"); sys.exit(subprocess.call(sys.argv[1:], env=env))

# ---- Toolchain ------------------------------------------------------------------
# Look for mise on PATH, then in the usual install locations (make started from a GUI may not
# inherit the shell's PATH). Override with make MISE=/path/to/mise.
# To try the behavior without mise, empty the candidates with MISE_CANDIDATES=.
MISE_CANDIDATES ?= $(HOME)/.local/bin/mise /opt/homebrew/bin/mise /usr/local/bin/mise
# Targets that do not run any tool from mise.toml (for example docker targets or targets that only
# print instructions). They still work when mise is not installed. List them on this line: the check
# below runs while make reads the file, so adding to the list later in the file has no effect.
NO_MISE_TARGETS := help
ifeq ($(SYSTEM_TOOLS),1)
RUN :=
else
ifndef MISE
MISE := $(firstword $(shell command -v mise 2>/dev/null) $(wildcard $(MISE_CANDIDATES)))
endif
ifeq ($(MISE),)
ifneq ($(filter-out $(NO_MISE_TARGETS),$(or $(MAKECMDGOALS),help)),)
$(error mise was not found. Install it from https://mise.jdx.dev, or add SYSTEM_TOOLS=1 to use the tools on PATH)
endif
endif
RUN := $(if $(MISE),$(MISE) exec --,)
endif

.PHONY: help setup build release run test lint fmt fmt-check check ci install uninstall clean smoke wheel pytest bench compile-check

## Setup

setup: ## ツールチェーン (mise) と依存を取得する
	@if [ -n "$(MISE)" ]; then "$(MISE)" install; fi
	$(RUN) cargo fetch $(CARGO_FLAGS)

## Build

build: ## デバッグ版をビルドする
	$(RUN) cargo build $(CARGO_FLAGS) --workspace

release: ## リリース版をビルドする
	$(RUN) cargo build --release $(CARGO_FLAGS) -p ocrus-cli

run: ## デバッグ版を実行する (引数は ARGS="...")
	$(RUN) cargo run $(CARGO_FLAGS) -p ocrus-cli -- $(ARGS)

## Checks

# Use default features so workspace tests link against Python; wheels enable extension-module.
test: ## テストを実行する
	@if [ "$$(uname -s)" = Darwin ]; then \
		$(RUN) $(PYTHON) -c '$(MACOS_CARGO_TEST)' cargo test $(CARGO_FLAGS) --workspace; \
	else \
		$(RUN) cargo test $(CARGO_FLAGS) --workspace; \
	fi

lint: ## clippy を警告ゼロで通す
	$(RUN) cargo clippy $(CARGO_FLAGS) --workspace --all-targets -- -D warnings

fmt: ## コードを整形する (書き換える)
	$(RUN) cargo fmt --all

fmt-check: ## 整形済みかを確かめる (書き換えない)
	$(RUN) cargo fmt --all -- --check

check: fmt-check lint compile-check ## 整形と静的検査 (書き換えない)

ci: check test ## CI と同じ検査 (書き換えない)

## Install

# Replace the binary through a temporary file and a rename instead of copying over it. macOS
# caches the code signature check per inode, so a binary copied over one that is running (or ran
# a moment ago) is killed with SIGKILL right after it starts (exit 137). The temporary file sits
# in the same directory so that the rename swaps the inode. The binary is not re-signed: the linker
# already signs it ad hoc, and a fixed identifier would not keep permissions across versions.
install: release ## リリース版を INSTALL_PATH (既定 /usr/local/bin) に入れる
	@mkdir -p "$(INSTALL_PATH)"
	@set -eu; \
		tmp=$$(mktemp "$(INSTALL_PATH)/.$(BINARY_NAME).XXXXXX"); \
		trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
		install -m 755 "target/release/$(BINARY_NAME)" "$$tmp"; \
		mv -f "$$tmp" "$(INSTALL_PATH)/$(BINARY_NAME)"

uninstall: ## INSTALL_PATH から取り除く
	rm -f "$(INSTALL_PATH)/$(BINARY_NAME)"

clean: ## ビルド成果物を消す
	$(RUN) cargo clean

## Additional checks and Python bindings

compile-check: ## Check every crate with default features
	$(RUN) cargo check $(CARGO_FLAGS) --workspace

smoke: ## Run the OCR smoke test (requires a model)
	$(RUN) cargo test $(CARGO_FLAGS) -p ocrus-engine --release --test smoke -- --nocapture

wheel: ## Build a Python wheel in target/wheels/
	cd python && $(RUN) maturin build --release --locked --out ../target/wheels

pytest: ## Test the installed Python wheel from the repository root
	$(RUN) $(PYTHON) -m pytest python/tests -q

bench: ## Run the benchmarks
	$(RUN) cargo bench $(CARGO_FLAGS)

## Help

help: ## このヘルプを表示する
	@echo "Development tasks for $(BINARY_NAME)"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Tool versions are pinned in mise.toml. Run make setup first."
	@echo "Release: GitHub Actions > Release > Run workflow"

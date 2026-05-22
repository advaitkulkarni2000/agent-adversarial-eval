# ── Agent Adversarial Eval — Makefile ────────────────────────
# Usage:
#   make run       → full pipeline (install + eval + results)
#   make install   → install dependencies only
#   make eval      → run evaluation only (assumes install done)
#   make clean     → remove generated files

.PHONY: run install eval clean

# ── Default target ────────────────────────────────────────────
run: install eval
	@echo ""
	@echo "✅ Done. Results saved to results/"
	@echo "   results/eval_results.png      — 4-panel chart"
	@echo "   results/raw_eval_results.csv  — per-prompt results"
	@echo "   results/metrics_summary.csv   — aggregated metrics"

# ── Install dependencies ──────────────────────────────────────
install:
	@echo "Installing dependencies..."
	pip install -q -r requirements.txt
	@echo "Dependencies installed."

# ── Run the evaluation ────────────────────────────────────────
eval:
	@echo ""
	@echo "Running agent adversarial evaluation..."
	@echo "This makes ~40 API calls (~8-12 minutes, ~$0.50-1.00)"
	@echo ""
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "ERROR: ANTHROPIC_API_KEY environment variable not set."; \
		echo "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."; \
		exit 1; \
	fi
	python run_eval.py

# ── Clean generated files ─────────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	rm -f agent_memory.db
	rm -f results/eval_results.png
	rm -f results/raw_eval_results.csv
	rm -f results/metrics_summary.csv
	@echo "Clean done. API key and notebook untouched."

# Psychophysics of concept injection in Gemma3-27B. GPU stages write raw CSVs to results/psychophysics/; CPU stages
# turn them into the summary tables and figures there.
#
#   make calibrate    α* per (layer, concept)
#   make sweep        every question at k·α*, concept and random vectors, all label schemes (1/2 on 30 passages,
#                     A/B and X/Y on 12, as in the committed results)
#   make summarize    raw CSVs → summary.csv, different/summary.csv, calibration_curve_concept.csv
#   make plot         every figure, from those tables (no GPU, no raw data needed)
#
# Outputs that already exist are skipped; pass FLAGS=--overwrite to redo them. `python psychophysics.py -h` lists the
# flags; FLAGS passes them through, e.g. make sweep FLAGS="--layers 31 --labels 12".

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

GPU   ?= 0
PY    ?= .venv/bin/python
FLAGS ?=

export CUDA_VISIBLE_DEVICES    := $(GPU)
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: help setup calibrate sweep summarize plot setup-figure selftest all

help:  ## list the targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## create .venv and install requirements.txt (uv)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

calibrate:  ## GPU: α* per (layer, concept) on the calibration passages
	$(PY) psychophysics.py calibrate $(FLAGS)

sweep:  ## GPU: every question at k·α* on the held-out passages (labels 1/2 on 30, A/B and X/Y on 12)
	$(PY) psychophysics.py sweep --labels 12 --passages 30 $(FLAGS)
	$(PY) psychophysics.py sweep --labels AB XY $(FLAGS)

summarize:  ## CPU: the raw CSVs → the summary tables
	$(PY) psychophysics.py summarize

plot:  ## CPU: every figure, from the summary tables
	$(PY) psychophysics.py plot

setup-figure:  ## CPU: the paradigm diagram
	$(PY) setup_figure.py

selftest:  ## GPU: cached-continuation scores must match a full forward pass
	$(PY) psychophysics.py selftest $(FLAGS)

all: calibrate sweep summarize plot  ## everything, in order

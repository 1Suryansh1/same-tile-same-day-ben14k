#!/bin/sh
python eval.py data/copernicusfm_vitb_frozen_valtest.npz && python rankers.py

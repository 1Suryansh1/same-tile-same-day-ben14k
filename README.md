# Same Tile, Same Day: Protocol Drift and a Pixel-Free Shortcut in BEN-14K Sensor-Agnostic Retrieval

Code and data for the paper by Suryansh Verma, Mayank Kumar, Shahzeb Ali and Dinesh Kumar Vishwakarma (Delhi Technological University), CV4EO Workshop at WACV 2027.

The paper audits BEN-14K, a benchmark for retrieval between Sentinel-1 radar and Sentinel-2 optical images. It shows that published results use different protocols, that a ranker using only patch location (no pixels) scores level with trained models, and proposes a strict score (the query's own acquisition removed from the gallery) and an acquisition holdout split. This repository lets anyone check a BEN-14K result against both.

## Contents

- `splits/ben14k_patches.csv`: all 14,832 BEN-14K pairs (Serbia, summer, official BigEarthNet v1 split): S2 and S1 patch IDs, v1 split, acquisition id (0 to 4, one tile and date), patch grid row/col, 19 class labels (BigEarthNet-19 names in the standard class order of `data/labels19_multihot.npy`).
- `splits/v1_{train,val,test}.txt`: S2 patch IDs per split (train 7,761, val 3,508, test 3,563).
- `splits/holdout_fold{0..4}.csv`: acquisition holdout folds. Fold f: test = acquisition f, val = acquisition (f+1) mod 5, train = the other three. No acquisition is shared between splits.
- `eval.py`: f1_hm (CSMAE definition), item F1, sklearn-style macro F1, strict score, any k, any query/gallery split, self match include/exclude, own-acquisition share, bootstrap CI over queries.
- `rankers.py`: random, acquisition only, location, band statistics.
- `data/copernicusfm_vitb_frozen_valtest.npz`: frozen Copernicus-FM ViT-B pooled embeddings (fp16) for val and test, both sensors, with patch IDs.
- `data/labels19_multihot.npy`, `data/bandstats_valtest.npz`: labels (manifest order) and band statistic features.

Requires Python 3 and NumPy only.

## Reproduce

    python eval.py data/copernicusfm_vitb_frozen_valtest.npz          # frozen Copernicus-FM row, standard and strict
    python eval.py data/copernicusfm_vitb_frozen_valtest.npz --boot 1000
    python eval.py data/copernicusfm_vitb_frozen_valtest.npz --query test --gallery test --k 5 --self_match include
    python rankers.py                                                  # trivial reference rankers

Expected (val->test, k=10, f1_hm, S1->S1 / S2->S2 / S1->S2 / S2->S1): standard 72.69 / 73.17 / 37.14 / 35.34, strict 66.57 / 60.50 / 36.81 / 34.58. The paper reports 72.69 / 73.16 / 37.13 / 35.35 and 66.58 / 60.49 / 36.81 / 34.57 from fp32 embeddings; the fp16 copy here differs by at most 0.01.

`rankers.py` prints random 39.73 / 33.96, acquisition 54.67 / 33.96, band statistics 70.94 / 54.13 (S2->S2) and 60.76 / 50.09 (S1->S1), standard / strict, and location about 72.7 / 33.4. The acquisition ranker here is the exact expectation; the paper averages 100 random permutations (54.67 / 33.94). The location ranker breaks distance ties with random jitter, so its last digits vary with the random stream and NumPy version (paper: 72.70 / 33.75). Under the strict score every remaining gallery item ties, so the location ranker orders them at random. The rankers without pixels score the same in every direction, since the S1 and S2 images of a pair share location, acquisition and labels.

## Reporting checklist for BEN-14K retrieval

1. Split: which patch lists (use `splits/`), and query set -> gallery set (val->test, test->val, test->test).
2. k (5, 10, ...).
3. F1 definition: f1_hm (harmonic mean of averaged P and R), item F1, or macro F1. Macro F1 counts absent labels; a random ranker scores about 62 with it.
4. For test->test: whether the query itself (or its paired other-sensor patch) is in the gallery.
5. Bands and preprocessing (band set, resampling of 20 m and 60 m bands, normalisation statistics).
6. Training data of any pretrained weights (for example BEN-270K contains BEN-14K images).
7. The strict score (own acquisition removed from the gallery) next to the standard score, and the own-acquisition share of the top 10.
8. Seeds and confidence intervals (bootstrap over queries captures query sampling only, not training variance).

## Data

BEN-14K is a subset of BigEarthNet-MM (Sumbul et al., 2021). The patch identifiers and labels here come from the official BigEarthNet release; please also cite BigEarthNet and CSMAE (Hackstein et al., 2025) when using them.

## Citation

    @inproceedings{verma2027sametile,
      title     = {Same Tile, Same Day: Protocol Drift and a Pixel-Free Shortcut in {BEN-14K} Sensor-Agnostic Retrieval},
      author    = {Verma, Suryansh and Kumar, Mayank and Ali, Shahzeb and Vishwakarma, Dinesh Kumar},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
      year      = {2027}
    }

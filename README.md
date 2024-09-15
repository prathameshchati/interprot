# pLM Interpretability

This repo contains tools used to interpret protein language models. `viz` contains the frontend app for visualizing SAE features. `plm_interpretability` is a python package containing tools for SAE training and interpretation.

## Running the visualizer

```bash
cd viz
pnpm install
pnpm run dev
```

## Running the auto-interpretation pipeline

### Step 1: Produce a labels CSV file

Download a DSSP (Dictionary of Secondary Structure in Proteins) file from [here](https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz) and place it in the `data/` directory.

Use the following command to produce a CSV file matching a desired secondary structure pattern (e.g. alpha helix: `H`, beta hairpin: `EEETTTEEEESSS`).

```bash
python3 -m plm_interpretability.autointerp pdb2labels --dssp-path plm_interpretability/autointerp/data/pdb_structure_annotations.txt --ss-patterns H --out-path plm_interpretability/autointerp/data/alpha_helix_labels.csv
```

### Step 2: Produce a CSV file that scores each SAE dimension on its ability to discriminate against the label

```bash
python3 -m plm_interpretability.autointerp labels2latents --labels-csv plm_interpretability/autointerp/data/alpha_helix_labels.csv --sae-checkpoint plm_interpretability/checkpoints/l24_plm1280_sae4096_k128_100k.pt --plm-dim 1280 --sae-dim 4096 --out-path plm_interpretability/autointerp/data/alpha_helix_mappings.csv
```

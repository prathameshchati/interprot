# pLM Interpretability

This repo contains tools used to interpret protein language models. `viz` contains the frontend app for visualizing SAE features. `plm_interpretability` is a python package containing tools for SAE training and interpretation.

## Running the visualizer

```bash
cd viz
pnpm install
pnpm run dev
```

## Running the auto-interpretation pipeline

### Step 1: Produce a classification CSV file

Download a DSSP (Dictionary of Secondary Structure in Proteins) file from [here](https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz) and place it in the `data/` directory.

Use the following command to produce a CSV file matching a desired secondary structure pattern (e.g. beta hairpin: `EEETTTEEEESSS`).

```bash
python3 -m plm_interpretability.autointerp pdb2class --dssp-path data/pdb_structure_annotations.txt --ss-patterns EEETTTEEEESSS --out-path data/beta_hairpin_class.csv
```

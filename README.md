# pLM Interpretability

This repo contains tools used to interpret protein language models. `viz` contains the frontend app for visualizing SAE features. `plm_interpretability` is a python package containing tools for SAE training and interpretation.

## Running the visualizer

```bash
cd viz
pnpm install
pnpm run dev
```

## Running the auto-interpretation pipeline

### Step 0: Docker setup

```bash
docker compose build
docker compose run --rm plm-interpretability bash
```

### Step 1: Produce a labels CSV file

Download a DSSP (Dictionary of Secondary Structure in Proteins) file from [here](https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz) and place it in the `data/` directory.

Use the following command to produce a CSV file matching a desired secondary structure pattern, e.g. beta hairpins encoded
by the regex `E{3,12}[T]{2,5}E{3,12}`.

```bash
autointerp pdb2labels --dssp-file plm_interpretability/autointerp/data/ss.txt --ss-patterns "E{3,12}[T]{2,5}E{3,12}" --out-path "plm_interpretability/autointerp/results/labels/E{3,12}[T]{2,5}E{3,12}_labels.csv"
```

### Step 2: Produce a CSV file that scores each SAE dimension on its ability to discriminate against the label

```bash
autointerp labels2latents --labels-csv "plm_interpretability/autointerp/results/labels/E{3,12}[T]{2,5}E{3,12}_labels.csv" --sae-checkpoint plm_interpretability/checkpoints/l24_plm1280_sae4096_k128_211k.pt --plm-dim 1280 --plm-layer 24 --sae-dim 4096 --out-path "plm_interpretability/autointerp/results/l24_plm1280_sae4096_k128_211k/E{3,12}[T]{2,5}E{3,12}_mapping.csv"
```

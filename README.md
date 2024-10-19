# pLM Interpretability

This repo contains tools used to interpret protein language models. `viz` contains the frontend app for visualizing SAE features. `plm_interpretability` is a python package containing tools for SAE training and interpretation.

## The visualizer

```bash
cd viz
pnpm install
pnpm run dev
```

## The python package

### Set up Docker container and run tests

```bash
docker compose build
docker compose run --rm plm-interpretability bash
pytest
```

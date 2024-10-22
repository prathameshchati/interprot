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

### Running commands

Each directory under `plm_interpretability` contains a command-line tool. For example, `make_viz_files` takes in an SAE checkpoint and generates JSON files containing SAE activations used to serve the visualizer. You can run it with

```bash
cd plm_interpretability
python -m make_viz_files \
    --checkpoint-files <path to checkpoint> \
    --output-dir <path to output directory where the JSON files will be saved>
```

Or you can install this repo as a package and run it from any directory:

```bash
RUN pip install -e .
make_viz_files --checkpoint-files <path to checkpoint> --output-dir <path to output>
```

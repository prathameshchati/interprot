# Binary classification probes

## Single latent

```bash
logistic_regression_probe single-latent \
--sae-checkpoint plm_interpretability/checkpoints/l24_plm1280_sae4096_k128_100k.pt \
--sae-dim 4096 \
--plm-dim 1280 \
--plm-layer 24 \
--swissprot-tsv plm_interpretability/logistic_regression_probe/data/swissprot.tsv \
--output-dir plm_interpretability/logistic_regression_probe/results \
--max-seqs-per-task 5 \
--annotation-names "DNA binding"
```

## All latents

```bash
logistic_regression_probe all-latents \
--sae-checkpoint plm_interpretability/checkpoints/l24_plm1280_sae4096_k128_100k.pt \
--sae-dim 4096 \
--plm-dim 1280 \
--plm-layer 24 \
--swissprot-tsv plm_interpretability/logistic_regression_probe/data/swissprot.tsv \
--output-file plm_interpretability/logistic_regression_probe/results/all_latents.csv \
--max-seqs-per-task 5 \
--annotation-names "DNA binding"
```

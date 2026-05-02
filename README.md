# friction-theory-p2-capacity-scaling

Companion repository for **Paper 2** of the friction-theory series:

> **Capacity Scaling of Encoding-Through-Loading: Application vs. Cloze Asymmetry Across Three Orders of Magnitude**
> Tomas Pødenphant Lund (2026). Independent Research, Aarhus.

This repository contains the per-token logprob datasets, analysis scripts, and manuscript that accompany the paper. It is the canonical "Data and code availability" reference cited in the paper.

## Contents

```
data/
├── precal_eval_all.jsonl          (52 MB) Colab Qwen2.5 0.5B → 14B sweep, all per-token logprobs
└── capacity_curve_together.jsonl  (2 MB)  Together/Fireworks 8B → 671B sweep with /no_think directive

scripts/
├── analyze_colab_capacity.py      Generates Figure 1 (capacity scaling curve) and the per-model summary table
└── validate_capacity_ceiling.py   Validates the active-vs-total parameter projection on MoE models (Figure 3)

manuscript/
├── paper2_capacity_scaling.md     Final manuscript (Markdown source; arXiv preprint linked below when live)
├── paper2_bibliography.md         Bibliography (mirrors §8 inline)
├── table1_per_model_summary.md    Per-model accuracy + CR statistics (referenced from §7)
└── figures/
    ├── fig1_capacity_scaling.png         Cloze + application accuracy vs log model size
    ├── fig2_bottleneck_migration.png     Per-fact outcome distribution across capacity
    └── fig3_moe_active_vs_total.png      Active-parameter projection vs total-parameter projection on MoE
```

## Headline findings

1. **Application scales monotonically with capacity** — Spearman ρ = +1.000 on the Qwen2.5 sub-ladder (n=5; p = 0.0083 one-tailed, 0.0167 two-tailed; slope +40.8 percentage points per decade); cross-family panel ρ = +0.92, n=9, p = 0.0005 (two-tailed)
2. **Cloze retrieval saturates early** — most models reach ~90% accuracy by 8B parameters
3. **Bottleneck migrates with capacity** — at 0.5B retrieval fails; at 14B retrieval is saturated and 36% of questions show "retrieval succeeds, derivation fails"
4. **Mixture-of-Experts (MoE) scales on active parameters, not total** — 235B MoE with 22B active behaves on application tasks like a 22B dense model

## Reproducing the analysis

```bash
# Generate Figure 1 + per-model summary
python scripts/analyze_colab_capacity.py

# Validate MoE active-parameter projection (Figure 3)
python scripts/validate_capacity_ceiling.py
```

Both scripts read directly from `data/*.jsonl`. No external API calls or model downloads required for re-analysis.

## Data format

Each line in the JSONL files is a single per-question record with at minimum:

- `model`: model identifier (e.g. `Qwen2.5-7B-Instruct`)
- `size_b`, `active_b`: total and active parameter counts in billions
- `question_id`, `question_type`: cloze | application
- `prompt`, `response`: full prompt and model output
- `correct`: ground-truth scoring
- `per_token_cr`: list of competing-routes values per generated token
- `raw_top_logprobs`: full top_logprobs array (top-5 alternatives per token)
- `mean_cr`, `std_cr`, `first5`, `last5`: aggregate friction statistics

The per-token logprob data is the most granular available; CR, entropy, and any future signals can be re-computed from `raw_top_logprobs` without re-running the evaluation.

## Citation

If you use this data or analysis, please cite:

```bibtex
@unpublished{lund2026capacity,
  author = {P{\o}denphant Lund, Tomas},
  title  = {Capacity Scaling of Encoding-Through-Loading:
            Application vs. Cloze Asymmetry Across Three Orders of Magnitude},
  year   = {2026},
  note   = {Manuscript in preparation. Companion repository:
            \url{https://github.com/tplund/friction-theory-p2-capacity-scaling}}
}
```

The arXiv preprint URL and Zenodo DOI will be added here when available.

## Companion papers

This repository is part of the friction-theory paper series:

- **Paper 1** — *Friction as the cost of probabilistic computation: A substrate-universal framework* (theoretical foundation)
- **Paper 2** — *this paper* (LLM-substrate test of the C-dimension prediction)
- **Paper 3** — *Friction-guided inference: a free signal that improves any LLM* (companion repo: [`friction-guided-inference`](https://github.com/tplund/friction-guided-inference))
- **Paper 4** — *Cross-substrate replication of classical learning phenomena on LLM substrate* (in preparation)

## License

All material in this repository is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to share and adapt with attribution.

## Contact

Tomas Pødenphant Lund — `tomas.lund@frictiontheory.org`
Web: [frictiontheory.org](https://frictiontheory.org)

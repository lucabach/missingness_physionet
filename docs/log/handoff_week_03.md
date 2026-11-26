# Week 3 handoff

GRU-D and ODE-RNN each completed 16 validation tuning runs and 48 final runs. Each task used eight tuning configurations. The final runs cover the three input variants, the mortality planted-pattern study and the comparable-variable sepsis contrast. Decisions D25 and D26 describe the recurrences. The GRU-D values variant is a plain GRU.

The table gives mean test AUROC across seeds 0, 1 and 2 for the full models. Sepsis uses pre-onset hours at hospital 1.

| Model | Mortality | Sepsis |
| --- | ---: | ---: |
| GRU-D | 0.872 | 0.755 |
| ODE-RNN | 0.862 | 0.757 |

Reproduce the sweeps and evaluations with [the Week 2 handoff, section 5.3](handoff_week_02.md#53-run-them-same-commands-as-the-gru). The frozen configurations are in `configs/experiment/`. The aggregate scores, paired bootstrap intervals and seed-level contrasts for Week 4 are in [runs.csv](../../results/runs.csv) and [measures.csv](../../results/measures.csv).

The figures use the same plotting script as the Week 2 handoff.

| Measure | GRU-D | ODE-RNN |
| --- | --- | --- |
| Information | [Figure](../../results/findings_grud_1_information.png) | [Figure](../../results/findings_odernn_1_information.png) |
| Removal | [Figure](../../results/findings_grud_2_removal.png) | [Figure](../../results/findings_odernn_2_removal.png) |
| Planted pattern | [Figure](../../results/findings_grud_3_planted.png) | [Figure](../../results/findings_odernn_3_planted.png) |
| Transfer | [Figure](../../results/findings_grud_4_transfer.png) | [Figure](../../results/findings_odernn_4_transfer.png) |

The separate [mortality](../../results/predictor_mortality.json) and [sepsis](../../results/predictor_sepsis.json) predictor reports record validation-selected weights, held-out metrics and the paired gain over the best individual checkpoint. See [predictor selection](../current/experiments.md#predictive-performance-across-seeds) for the selection rule and inference instructions.

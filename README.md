# Uncertainty-Calibrated Fault Diagnosis with XAI-Grounded LLM Explanations

Source code, results and figures for the article *Uncertainty-Calibrated Fault Diagnosis with XAI-Grounded LLM Explanations* (Ilkay Yelmen, Metin Zontul, Halit Bakir and Seda Karateke), submitted to *Sensors*.

The pipeline compares five classifiers (XGBoost, 1D-CNN, KAN, LSTM, Logistic Regression) on the Tennessee Eastman Process (TEP) and Case Western Reserve University (CWRU) bearing datasets, calibrates the 1D-CNN with temperature scaling, evaluates selective prediction and robustness to sensor degradation, explains each prediction with Integrated Gradients (IG), and passes the calibrated confidence and the attribution evidence to four large language models (GPT-5 Mini, Qwen3.5-9B, DeepSeek-V4.1-Flash-1017, GLM-5.3-Flash) that write the diagnostic explanation. The explanations are scored for Evidence Adherence@1, Recall Faithfulness and hallucination; the Explanation Reliability Score (ERS) is evaluated as an attribution audit measure.

## Repository layout

```
Source Code/
├── paths.py               data and output locations
├── tep_data.py            TEP loading, parquet cache, windowing, sensor degradation
├── cwru_data.py           CWRU loading, noise injection, CWRU prompt domain
├── models.py              1D-CNN, trainer, temperature scaling, Integrated Gradients
├── llm.py                 LLM clients, prompts, response schema, metrics, ERS
├── stat_utils.py          bootstrap, calibration and significance helpers
├── train_classifiers.py   TEP 1D-CNN training and five-classifier comparison
├── calibration.py         calibration outputs under clean and noisy inputs
├── cwru_experiments.py    CWRU pipeline (stages 1-5)
├── xai_evaluation.py      attribution reliability, deletion, robustness, ERS tests
├── llm_experiment.py      LLM explanation experiment and re-scoring
├── llm_analysis.py        LLM tables, statistics and Figure 4
├── llm_revision.py        final LLM runs, ablations, negative controls, ERS baselines, physical-reference check, consolidation
├── validation_round1.py   fault-onset correction, XGBoost statistics, DEFE, sensitivity checks
├── validation_round2.py   leakage-free CWRU CV, label consistency, IG convergence, held-out cases
├── figures.py             Figures 1-3
├── resources/             physical reference sensor sets
├── figures/               figures as published in the article
├── results/               result tables behind the article (CSV/JSON)
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.12 or newer (tested with Python 3.13; `llm.py` uses nested same-quote f-strings, which Python 3.11 does not accept). `git` is needed because `requirements.txt` installs `efficient-kan` (the KAN classifier) from GitHub.

```
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from pytorch.org if a GPU is used. The experiments were run with PyTorch 2.7.1 (CUDA 11.8).

## Data

Download the datasets and place them under `data/` (or set `FD_DATA_DIR`):

| Dataset | Folder | Source |
|---|---|---|
| Tennessee Eastman Process (Rieth et al.) | `data/TEP/` (`TEP_FaultFree_Training.RData`, `TEP_FaultFree_Testing.RData`, `TEP_Faulty_Training.RData`, `TEP_Faulty_Testing.RData`) | Harvard Dataverse, doi:10.7910/DVN/6C3JR1 |
| CWRU bearing data, 12 kHz drive end, 0 hp | `data/CWRU/` (`97.mat`, `105.mat`, `118.mat`, `130.mat`, `169.mat`, `185.mat`, `197.mat`, `209.mat`, `222.mat`, `234.mat`) | Case Western Reserve University Bearing Data Center |
| CWRU bearing data, 12 kHz drive end, 1-3 hp (cross-load and held-out cases) | `data/CWRU_loads/` (1 hp: `98.mat`, `106.mat`, `119.mat`, `131.mat`, `170.mat`, `186.mat`, `198.mat`, `210.mat`, `223.mat`, `235.mat`; 2 hp: `99.mat` ... `236.mat`; 3 hp: `100.mat` ... `237.mat`, same class order as the 0 hp files) | Case Western Reserve University Bearing Data Center |

All generated files (models, caches, cases, LLM responses, tables, figures) are written to `outputs/` (or `FD_OUT_DIR`).

## LLM access

The results reported in the article were obtained through hosted APIs. GPT-5 Mini (served model `gpt-5-mini-2025-08-07`) was called through the OpenAI API. Qwen3.5-9B, GLM-5.3-Flash and DeepSeek-V4.1-Flash-1017 were called through Together AI's serverless OpenAI-compatible API (`https://api.together.xyz/v1`, the default in `llm.py`).

Call settings: temperature 0.0 and a maximum of 8,192 output tokens for the Together AI models; for GPT-5 Mini the temperature parameter is not sent (provider default) and the same 8,192-token limit is applied. When a reasoning model reached the limit without a final answer, the call was repeated with a 32,768-token limit (only Qwen3.5-9B needed this). Failed requests were retried up to six times, honouring the `Retry-After` header or using exponential back-off (5-120 s).

API keys are read from environment variables only and are not part of this repository:

```
LLM_API_KEY        OpenAI API key (GPT-5 Mini)
OPEN_LLM_API_KEY   Together AI API key (open-weight models); TOGETHER_API_KEY is also accepted
OPEN_LLM_BASE      optional; overrides the endpoint of the open-weight models
```

Any other OpenAI-compatible endpoint can be used by setting `OPEN_LLM_BASE`, but results obtained that way are not guaranteed to match the reported ones, because hosted services may change or withdraw model versions (two initially selected models became unavailable on Together AI during the revision and were replaced).

## Running

Each script prints its options with `-h`. Scripts with several parts use subcommands, e.g. `python xai_evaluation.py deletion -h`.

1. TEP cache and classifiers
   ```
   python tep_data.py --convert
   python train_classifiers.py final
   python train_classifiers.py tep
   python train_classifiers.py ablation
   python calibration.py
   ```
   `final` retrains the 1D-CNN with a simple train/test split, `tep` trains it with run-grouped validation and early stopping, and `ablation` runs the five-classifier comparison (five seeds, five run-grouped folds).
2. CWRU pipeline (stage 1 prepare, 2 train, 3 calibrate, 4 integrated gradients, 5 LLM explanations)
   ```
   python cwru_experiments.py
   python cwru_experiments.py --stage 5
   ```
3. Explainability
   ```
   python xai_evaluation.py reliability --dataset tep
   python xai_evaluation.py reliability --dataset cwru
   python xai_evaluation.py deletion
   python xai_evaluation.py robustness
   python xai_evaluation.py agar
   python xai_evaluation.py stability
   python xai_evaluation.py spcompare
   python xai_evaluation.py ers
   python xai_evaluation.py ersablation
   ```
   `reliability` also takes `--snr-db` (CWRU degraded windows) and `--checkpoint`, `--cases` and `--output` to select the inputs of the noisy-input runs.
4. LLM explanations (requires the API keys above)
   ```
   python llm_experiment.py run --help
   python llm_experiment.py run --models mock --mock-cases
   python llm_revision.py registry
   python llm_revision.py manifest
   python llm_revision.py run --dataset TEP
   python llm_revision.py run --dataset CWRU
   python llm_revision.py ablation-run --dataset tep --models gpt5_mini qwen35_9b
   python llm_revision.py final
   python llm_revision.py run-details
   python llm_revision.py ablation
   python llm_revision.py controls
   python llm_revision.py ers-baseline
   python llm_revision.py ers-final
   python llm_revision.py physical
   python llm_analysis.py tables
   python llm_analysis.py stats
   python llm_analysis.py table2
   ```
   `llm_experiment.py run --models mock --mock-cases` exercises the pipeline offline with synthetic output and needs no API key. `llm_revision.py registry` prints the four model keys (`gpt5_mini`, `qwen35_9b`, `deepseek_v41_flash`, `glm53_flash`) with their model ids. The revision runs are listed in `RUNS` in `llm_revision.py`: retry of failed requests, CWRU clean rerun, the two replacement models (generate their request lists with `newmodel-manifest --models <key>` and pass the file to `run --models <key> --manifest <file>`), and the Rank Only / Percentages Only ablation (`ablation-run`, one dataset and model set at a time). `final` consolidates all blocks into the 1,296 records behind the article.
5. Validation analyses (`-h` lists the subcommands of each script)
   ```
   python validation_round1.py onset
   python validation_round1.py ers
   python validation_round1.py ba-shortcut
   python validation_round1.py fold-stats
   python validation_round1.py defe
   python validation_round1.py tables
   python validation_round1.py xai-sens
   python validation_round1.py xgb-rerun
   python validation_round2.py e1
   python validation_round2.py e2
   python validation_round2.py e2b
   python validation_round2.py e3
   python validation_round2.py e4
   python validation_round2.py cases
   python validation_round2.py llm-manifest
   python validation_round2.py llm-run --exp <experiment> --dataset <dataset>
   python validation_round2.py analyse
   ```
   `validation_round1.py` covers the fault-onset correction, XGBoost statistics (`xgb-rerun`, `xgb-analyse`, `xgb-raw5`, `xgb-analyse-raw5`), DEFE and sensitivity checks. `validation_round2.py` covers the leakage-free CWRU cross-validation, TEP label consistency, IG convergence and the 40 held-out cases (`cases`, then `llm-manifest` and `llm-run` for the LLM calls).
6. Figures
   ```
   python figures.py fig1
   python figures.py fig2
   python figures.py fig3
   python llm_analysis.py tables
   python validation_round2.py analyse per_case
   ```

## Figures

`figures/` holds the figures as published. The scripts write their own copies to `outputs/figures/` under the names in the last column; the files in `figures/` are renamed copies of them.

| File in `figures/` | Article | Produced by | Script output in `outputs/figures/` |
|---|---|---|---|
| `Figure_1_TEP_Classifier_Comparison.png` | Figure 1 | `figures.py fig1` | `Figure1_TEP_Publication_Final.*` |
| `Figure_2_TEP_Calibration_Reliability.png` | Figure 2 | `figures.py fig2` | `Figure2_TEP_Calibration_Final.*` |
| `Figure_3_XAI_Heatmap_TEP_Fault4.png` | Figure 3 | `figures.py fig3` | `Figure3_XAI_Heatmap.png` |
| `Figure_4_Explanation_Alignment.png` | Figure 4 | `llm_analysis.py tables` | `Figure4_Explanation_Alignment.png` |
| `Figure_S1_Per_Case_EA1.png` | Figure S1 | `validation_round2.py analyse per_case` | `Figure_R2_1_per_case_EA1.png` |

## Results

`results/` holds the numbers reported in the article, so they can be checked without re-running the pipeline.

| Folder | Content |
|---|---|
| `classifiers/` | TEP and CWRU classifier comparison, fold-level results, XGBoost statistics, fault-onset and shortcut checks, DEFE features |
| `calibration/` | temperature-scaling calibration, selective prediction, reliability-diagram bins, statistical tests |
| `xai/` | deletion, robustness, AGAR and stability-profile tests, ERS component ablation, attribution reliability per noise level |
| `cases/` | diagnostic cases (prediction, calibrated confidence, attribution evidence) sent to the LLMs |
| `llm/` | all 1,296 LLM records (`final_results.csv`: 1,232 model responses and 64 system-level abstentions, four models), explanation-alignment tables, model comparison, degradation, ERS, rank/percentage ablation, negative controls, physical-reference check |
| `validation/` | leakage-free CWRU cross-validation, cross-load test, TEP label consistency, IG convergence, shuffled-attribution ablation, held-out cases, per-case EA@1 |

## License

MIT License, see `LICENSE`.

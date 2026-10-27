# Quant ML Futures Academy

A free, static, mobile-friendly **12-week course** focused on researching and validating machine-learning-assisted ES/NQ futures strategies with **NinjaTrader 8 + Python + NVIDIA RTX 5070 Ti**.

## Run it now

Open `index.html` with Chrome, Safari, or Edge, or publish these files to GitHub Pages. No server, database, API keys, login, or JavaScript build step is needed. The Google Fonts import (optional) and linked videos require internet. Course text, quizzes, labs, flashcards, and status are in the site code.

### Publish using GitHub Pages (no terminal needed)

1. On GitHub click **New repository** and name it `quant-ml-futures-academy`. Choose **Public** to make the site easy to share.
2. **Upload all website files into the repository root**, especially `index.html`, `styles.css`, `app.js`, `data.js`, `.nojekyll`, and `gpu_check.py`. Do not upload the containing folder as a nested directory unless you configure Pages to serve that directory.
3. Commit the files to the default `main` branch.
4. Go to **Settings → Pages → Build and deployment → Deploy from a branch**; choose `main` and `/(root)`; click **Save**.
5. When GitHub displays the live address, share `https://YOUR-USERNAME.github.io/quant-ml-futures-academy/`.
6. On iPhone in Safari choose Share → Add to Home Screen; on Android in Chrome choose ⋮ → Add to Home screen. A home-screen shortcut is not the same as a full installed offline Progressive Web App.

If you already have a repo, you can use that instead without changing the course source. Files are static. GitHub Pages availability and publishing options depend on repository/plan configuration.

## What's in the course

- 12 modules covering hypothesis testing, statistics, futures order flow, data hygiene, execution costs, event-based features/labels, logistic and tree baselines, purged forward validation, RTX/PyTorch, regimes, NT8 integration, and final capstone.
- 48 graded multiple-choice questions with explanations, 36 flashcards, weekly build assignments, notes, completion tracking.
- 5 simulation tools: trading expectancy, ES/NQ tick math, economic cutoff, walk-forward visualization, and fail-closed NT8 signal gate.
- 9-item capstone readiness checklist. Training videos/lessons link to CME Group, Google MLCC, MIT OCW, StatQuest, NinjaTrader, NVIDIA and PyTorch.
- Per-device `localStorage` tracking. Export progress to JSON and import on other devices. No cloud sync or accounts.

## GPU setup

The RTX 5070 Ti is Blackwell compute capability 12.0 with 16 GB of VRAM. Update your NVIDIA driver. On Windows PowerShell:

```powershell
nvidia-smi
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy pandas pyarrow scikit-learn matplotlib jupyter
```

Install PyTorch using the **current CUDA-enabled Windows/Pip selector at https://pytorch.org/get-started/locally/**. This avoids an outdated wheel that may not support compute capability 12.0.

```powershell
python gpu_check.py
```

`gpu_check.py` prints the CUDA build/device and runs an actual GPU matrix multiplication.

## Starter supervised research script

`trade_event_baseline.py` is an educational **CPU** baseline that reads an actual event-level CSV of eligible trades and evaluates logistic vs histogram gradient boosting models against take-all and fixed trend-filter baselines, with chronological split and temporal label purging.

```powershell
python trade_event_baseline.py --csv path\to\my_trade_events.csv --instrument ES
```

Expected columns:

| column | Meaning |
| --- | --- |
| `event_time` | ISO timestamp with timezone when decision is made |
| `label_end_time` | ISO timestamp with timezone when the future-based label finishes |
| `instrument` | ES or NQ |
| `direction` | Long or Short |
| `atr7`, `atr21`, `adx21` | Past-only indicators, known at event time |
| `box_width_atr` | Box width normalized by past ATR |
| `prebreak_volume_ratio` | Volume ratio computed entirely before entry |
| `trend_alignment` | -1 / 0 / +1 based on directionally aligned past EMAs |
| `hour_ct` | Numeric exchange-local RTH hour at decision |
| `baseline_net_pnl` | Fixed-rule simulated trade PnL **after** transaction costs |

**Important:** This is a starter split, not full purged multi-fold walk-forward validation; the validation-chosen threshold is subject to selection bias. It is for training and exploratory comparisons only. No order placement is included.

## Safety and honest limits

- Futures are leveraged. One ES/NQ contract can incur substantial losses. Results and code are **educational**, not profitable-strategy claims.
- Market Replay with Level II is not the same as plain historical last-tick data. NinjaScript `OnMarketDepth()` is live only in ordinary strategy historical backtests.
- Tick Replay does **not** guarantee realistic fills. Incorporate spread, commissions, stop/limit queue uncertainty, slippage and stress assumptions.
- Never use a model to override independently enforced stops, one-position limits, session/overnight gates, daily loss limits or a kill switch.
- The hosted app is a public *curriculum*, **not** a trading service; never put API keys, brokerage credentials, paid raw market data or private PnL into the public GitHub repo.
- Import/export progress is not encrypted; avoid recording secrets in your notes.

## Sources / further study

- NVIDIA GPU capability https://developer.nvidia.com/cuda/gpus
- PyTorch setup https://pytorch.org/get-started/locally/
- PyTorch intro videos https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html
- NinjaScript documentation https://docs.ninjatrader.com/ninjascript
- NT8 L2 callback https://docs.ninjatrader.com/ninjascript/onmarketdepth
- NT8 Tick Replay cautions https://docs.ninjatrader.com/ninjascript/developing_for_tick_replay
- CME Introduction to Futures https://www.cmegroup.com/education/courses/introduction-to-futures
- Google ML Crash Course https://developers.google.com/machine-learning/crash-course/
- MIT probability/statistics https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/
- StatQuest https://www.youtube.com/@statquest

Free reference links may change over time. The course does not reproduce copyrighted lectures or videos. This project is independent of CME, NinjaTrader, NVIDIA, Google, MIT, or PyTorch.
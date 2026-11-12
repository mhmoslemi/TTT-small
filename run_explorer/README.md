# Run Explorer

Turns a training-run directory into **one self-contained, interactive HTML file**.
Nothing in the code base or in the run directory is modified - everything lives in this folder.

**Problems supported out of the box:** `ac1` (first autocorrelation inequality) and `erdos` (Erdős minimum-overlap constant C₅). The problem is read from the run's `config.json`;
the explorer then switches wording, the construction plots (autoconvolution f*f vs. overlap profile C(s)), the gap scale, the reference values and the Published SOTA tab accordingly.
It also copes with older runs (no JEV gate, no `best.json`, missing allocation summaries - these are rebuilt from the rollout files).

```bash
cd run_explorer
./make_explorer.sh ../runs/ac1_Qwen3.8-27B_1108-1201-step3 --open     # -> out/ac1_….html  (≈10 MB)
./make_explorer.sh ../runs/erdos_Qwen3.8-27B_1013-0544 --open         # -> out/erdos_….html (≈3 MB)
./make_explorer.sh --unified --open                                    # -> out/all_problems.html (≈13 MB): ONE page with every problem
# all files work offline - double-click to open
```

`make_explorer.sh` options (quote run names that contain spaces):

| usage | effect |
|---|---|
| `./make_explorer.sh RUN_DIR` | build `out/<run name>.html` |
| `./make_explorer.sh RUN_A RUN_B` | build several runs |
| `./make_explorer.sh` | use the most recently modified run under `../runs` |
| `--open` | open the result in the browser |
| `--unified [RUN_DIR ...]` | build **one** HTML with all the given runs (different problems): landing page + the full explorer of each run. Without RUN_DIRs the runs listed in `unified_runs.txt` are used. Output `out/all_problems.html` |
| `-o FILE` | choose the output path (single run, or the unified page) |
| `--all-code` | embed the source of every valid program (≈ 25-35 MB) |
| `--code-per-step N` | embed source for the N best valid programs per step (default 25) |

Needs only Python ≥ 3.8 (standard library; set `PYTHON=/path/to/python3` to choose another) and a current browser
(Chrome / Edge / Safari 16.4+ / Firefox). Building takes a few seconds. Re-run it any time the run has progressed -
the HTML is a snapshot of the directory at build time. (`python3 build_explorer.py RUN_DIR` does the same without the shell wrapper.)

## Unified page (all problems in one HTML)
`./make_explorer.sh --unified RUN_AC1 RUN_ERDOS` (or just `--unified`, which reads `unified_runs.txt`) writes one file with

* **All problems** (landing): one card per run - best value vs the problem's reference, gap in absolute and %, rank among the published results, best-so-far curve with the reference line, models, selection, valid %, wall clock and GPU-hours, **GPU specification**;
  a normalised **"progress toward the published reference"** chart (0 % = best of the first step, 100 % = the reference) that puts different problems on one axis; a side-by-side table and a **Hardware used** table (card, memory, compute capability, vLLM engines / tensor-parallel, evaluation CPUs).
* A pill per problem in the top bar that opens that run's complete explorer (all tabs, what-ifs, constructions ...). Every explorer lives in its own frame, is inflated only when first opened, and keeps its state while you switch between problems.
  The theme button is shared; `↗ new tab` opens the current explorer on its own.
* Add a problem by adding its run directory to `unified_runs.txt` (or on the command line) - problems without a profile still get the landing card and the pipeline tabs.

## Tabs

| Tab | What you can do |
|---|---|
| **Overview** | **hardware used (GPU card, memory, vLLM layout, GPU-hours)**, best-so-far vs a reference, **league table against published AC1 results**, failure breakdown, score distribution, wall-clock per phase, per-step table |
| **Pipeline (one step)** | the 8 stages with real counts; **animated flow diagram** (parents → plan → budget → result → training signal); click a stage for mechanism, formula, code references and live data; ▶ walkthrough |
| **Search: PUCT + JEV** | PUCT components with a live *c* slider; **JEV on the search tree** (the scanned candidates as tree nodes labelled with their upper bound, accept / reject recoloured live as you move the threshold, rejection arcs with the similarity, a zoomed upper-bound strip, and threshold sweeps); JEV comparison matrix with an **animated scan**; **genealogy that grows step by step**; score landscape |
| **Strategist** | every strategy slot (attempts / fallbacks), the chain of any parent, plan text, reasoning, prompt anatomy, what the coder did with it |
| **Allocation (hurdle)** | pilot outcomes vs the hurdle, the allocator's decision, a Monte-Carlo replica (agrees with the run on 96 % of parents) |
| **SPO² & training** | advantage function with β / baseline sliders, SPO² vs entropic baseline, tracker simulation (reproduces the logged baseline to 1e-10), PPO-clip plot, loss / gradient / timing curves |
| **Rollouts explorer** | filter + scatter + sortable table; click any program for code, strategy, stdout, construction plot, **diff vs parent**, advantage derivation |
| **Solution (AC1)** | step function and autoconvolution recomputed in the browser; compare any two constructions (this run's best after each step **or the published ones**: AlphaEvolve, ThetaEvolve, TTT-Discover, Together AI, SimpleTES, best human); ▶ morph through the steps; **step function and f*f after every step** (small multiples) plus an overlay of all steps; sequence lengths |
| **Published SOTA** | record progression over time, every published AC1 result with the bound **recomputed from its public height sequence**, the verbatim Table 17 of arXiv:2604.19341 (AC1/AC2/AC3), what happened in order, source links, and clearly flagged unreviewed claims |
| **How it works** | architecture diagram, stage reference with formulas and `file:line` pointers, glossary, "easy to get wrong" list, file map |
| **Config & event log** | config.json, startup banner, filtered terminal events |

### Hardware
The explorer reports the GPUs the run used: count and ids (from `config.json`), memory and compute capability (vLLM start-up log), vLLM engines / tensor-parallel degree, evaluation CPUs and GPU-hours.
The logs never print the card's *name*, so it is taken from `hardware_profile` when the run has one (AC1: `B200_full_ram` → 2 × B200, 178 GiB) and otherwise **inferred**
(the Erdős run predates that setting: capability 12.0 and 95 GiB per GPU match the repo's default RTX6000 profile → 8 × RTX PRO 6000 Blackwell). Inferred names carry a "model inferred" tag and the basis is shown in the Hardware card.

### Linked what-ifs
All parameters live in one shared store, so a change made anywhere shows up everywhere:
`β`, `d_half`, `ρ_min/ρ_max`, the per-step baseline override and `ε_low/ε_high` (SPO² tab ⇄ Pipeline stage 6/7 ⇄ flow diagram ⇄ Rollouts advantages),
the PUCT constant `c` (Search ⇄ Pipeline stage 1), the JEV threshold (Search replay and rejection chart ⇄ Pipeline stage 1) and the **Reference** value (every gap, line and the league table).
A `↺ reset what-ifs` button appears in the top bar while anything differs from the logged run.

### Animation
Charts draw in when a tab opens; the Pipeline flow reveals left → right and lights up the matching stage; the JEV scan can be animated candidate by candidate;
**▶ play steps** (top bar, or the space bar) plays through the run - the genealogy grows and every tab follows the step. Honours `prefers-reduced-motion`.

### Results key (published works only)
The top-bar **Results** key decides which published results are visible *everywhere* (reference list, league table, Published SOTA tab, Solution tab, unified page):

| level | shows |
|---|---|
| **published works only** (default) | papers / arXiv preprints with a public result (incl. the best human bound, AlphaEvolve, ThetaEvolve, TTT-Discover, Together AI, SimpleTES and the Einstein Arena paper) |
| + Arena leaderboard | adds the replayed Einstein Arena / community-list entries (AgentDiscover, Atris, CodexProLong ...) - not papers |
| + unreviewed claims | adds unreviewed claims (MILO, the Cumes preprint ...) - flagged, never ranked |

Each result carries a `tier` (`paper` / `leaderboard` / `claim`) in `data/published_<problem>.json`; events, tables and links are tiered too. The setting is remembered, and in the unified page it is shared with every explorer (ranks and gaps on the cards follow it).

### Reference values
The reference line / gap defaults to the record of the problem's reference table: for **AC1** the best bound in *arXiv:2604.19341v2, Supplementary Table 17* (Together AI, 1.502862), for **Erdős** the bound that paper presents as the new record (SimpleTES, 0.380868).
Everything else is selectable in the top-bar **Reference** dropdown: the other published methods, newer results from the community-curated lists (teorth/optimizationproblems, constants 1a / 1b - independently replayed entries, e.g. AC1 1.5027435 and Erdős 0.38085855), the Einstein Arena paper (arXiv:2606.10402), unreviewed claims (flagged, never ranked) and the run's own prompt target.

### Published constructions
`data/published_<problem>.json` ships with the explorer (AC1 ≈ 1 MB, Erdős ≈ 50 kB): the height sequences of the best human bound, AlphaEvolve, ThetaEvolve, TTT-Discover, Together AI, SimpleTES, Haugland ...,
downloaded from pinned git commits (URLs and SHA-256 are in the file) by `data/build_published.py`. The explorer recomputes every bound from these sequences in the browser
(FFT for the 30,000-step ones). All of them agree with the exact big-integer replays of teorth/optimizationproblems#348 to ~1e-9 (AC1) / ~1e-12 (Erdős), and this run's own best constructions reproduce their logged scores to ~1e-10.
Re-run `python3 data/build_published.py [ac1|erdos]` (needs internet) to refresh them.

### Adding another problem
1. In `template/js/problems.js` add a profile (name, metric, decimals, gap scale, `prep(vals)` returning the step function and the profile curve + its maximum, and the explanatory texts).
2. Optionally create `data/published_<problem>.json` (same layout as the existing files) for the SOTA tab and the reference lines.
Without a profile the explorer still works (pipeline, search, strategist, training, rollouts) and just skips the construction plots.

## Caveats worth knowing
* For runs with a JEV gate, `config.json` is stale for the threshold: the real value is stored per step (the AC1 run used 0.9 for steps 0-3 and 0.68 from step 4).
* Wall-clock phases are reconstructed from the HH:MM:SS timestamps of `temirnal.log` (no dates; resumed steps include restart time).
* The allocation sandbox samples the allocator's model instead of integrating it, so expected-improvement numbers differ slightly.
* The tracker simulation holds the measured policy shift D fixed when you change β.
* The partially written step shows only the stages that already wrote files.
* **JEV substitutes.** `jev.json` only stores the candidates GLiDE actually compared (the scan stops after the 8th acceptance), but lowering the threshold would push the scan further down the PUCT ranking. `archive_sim.py` therefore rebuilds the candidate window (top 24 states) of every step by replaying the rules of `search/sampler.py` on the saved rollouts (exact-code dedupe, top-2 per parent×strategy, top-8 per parent, rank prior, visit counts). It is checked against the run: the rebuilt archive size equals the logged `archive_size` at every step and the rebuilt scan reproduces the recorded one (programs with exactly equal reward may swap places, which never changes who is selected). In the JEV tree view the programs that would fill the vacated slots are shown as **substitutes** (blue, marked *): GLiDE never compared them, so they are assumed to pass. If a step's window cannot be validated the view falls back to the record-only replay.
* A step writes `summary.json` / `best.json` / `result.txt` only after its adapter update, but `temirnal.log` already prints `[step N] best-ever raw …` when its rollouts are scored. Such a step is shown as **"rollouts done"**: its programs, best value and construction count for best-so-far (Overview, Solution, Published SOTA, unified page, cross-checked against the logged value), while training / archive numbers stay "pending" until the step finishes. Steps still generating rollouts are listed as in progress and do not move the best.
* Baselines, advantages and the Throughput numbers use the logged values until you move a what-if control; a log that ends in a traceback / KeyboardInterrupt is reported as "stopped" and the finish estimate is replaced by "time left if resumed".

## Layout
```
make_explorer.sh    shell launcher (run directory in, html out)
build_explorer.py   CLI: collect -> pack -> inline into the template
collect.py          read-only reader of the run directory (parallel per step)
archive_sim.py      rebuilds the PUCT candidate window of every JEV step (for the substitutes) by replaying search/sampler.py's rules
data/               published_<problem>.json (reference constructions, tables, history) and the script that rebuilds them
unified_runs.txt    runs used by --unified when none are given
template/           index.html, style.css, js/*.js  (vanilla JS, SVG/canvas charts, no dependencies; problems.js = per-problem profiles)
                    hub.html / hub.css / hub.js = the unified page (landing + iframes)
out/                generated HTML
```

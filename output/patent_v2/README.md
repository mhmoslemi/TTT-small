# Patent presentation

Open **patent-v2.html** in a browser. It is self-contained, works offline, and needs no PowerPoint.

- Slides open fully visible so they can be read without a presenter. **Play** replays the animated builds.
- **Right / Space**: next slide (or next build while an animation is paused).
- **Left**: previous build or slide.
- **Page Down / Page Up**: change slide directly.
- **F**: full screen. **N**: speaker notes and sources. **O**: slide overview.
- **Play** animates the current pipeline, allocation, or scheduling slide.

**patent-v2.pdf** is the complete static version. **allocation-walkthrough.gif** is the allocation animation for reuse elsewhere. The PDF shows all animation builds at once.

**patent-v2.pptx** contains 30 static slides with editable titles, body text, footnotes, page numbers, and figure labels. The PDF contains 19 pages. Original pages 7, 9, and 10 expand into successive builds, retaining the same printed page number within each sequence. Advance manually through each build. The PPTX contains no GIFs or automatic animations. Equations, plot geometry, table rules, and the logo remain graphics. Text boxes use the original fonts and measured positions, with existing line breaks preserved.

The presentation includes the technical solution, an observed allocation, saved construction evolution, and the evaluated scientific result. Huawei branding and the supplied slide style are preserved. Display equations use native MathML typesetting with STIX Two Math.

The slides contain the technical explanation, equation definitions, and figure labels. Source details and comparison limitations are retained in speaker notes and **evidence.json**.

The Qwen3-8B ablation uses the saved measurement artifact, not a new experiment. It matches coder attempts, not total compute. The scientific result uses the exact saved 0544 construction and score; it is not a refreshed state-of-the-art or patentability assessment. The supplied inventor declarations require inventor confirmation before submission.

Three additional evidence pages connect experimental observations to design choices: coder learning and reward design, planning and coding models, and two runs of the full pipeline. Historical report snapshots are transcribed in **report-experiments.json**, with source pages and their scope. The build derives the system trajectories from saved run metadata. Unequal-budget component comparisons are not presented as controlled causal ablations. Report snapshots are not silently mixed with later final run totals.

Construction figures use the same score-preserving orientation as `plot_erdos.py`. The embedded evidence retains each raw saved array alongside its display-oriented array. Scheduling colors distinguish the planner, coder, evaluation, allocation, and training roles.

For text or diagram edits, change **slides.template.html** or **deck.js**, then run `node output/patent_v2/build.mjs` from the repository root. This rebuilds the standalone HTML and its **evidence.json** provenance record. The build uses saved JSON only and does not import training code, execute generated programs, contact a server, or change run artifacts.

To rebuild the PPTX, run `render.mjs PRIVATE_BUILD_DIR --editable-pptx-assets` after the HTML build, then `export-pptx.mjs PRIVATE_BUILD_DIR`. Both scripts live in this directory. The renderer extracts visible text and positions from the HTML and captures the graphics with text removed. The exporter writes native PowerPoint text boxes and expands complete animation builds into static slides.

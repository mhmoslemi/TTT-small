# Patent presentation

Open **patent-v2.html** in a browser. It is self-contained, works offline, and needs no PowerPoint.

- Slides open fully visible so they can be read without a presenter. **Play** replays the animated builds.
- **Right / Space**: next slide (or next build while an animation is paused).
- **Left**: previous build or slide.
- **Page Down / Page Up**: change slide directly.
- **F**: full screen. **N**: speaker notes and sources. **O**: slide overview.
- **Play** animates the current pipeline, allocation, or scheduling slide.

**framework-animation.html** opens the framework sequence directly and plays offline. Use Play / Pause or Replay to control it. **framework-animation.mp4** is its 27-second video export, and **framework-animation.gif** is a looping preview. The six click-through builds on printed page 7 use the same diagram. Its shapes, paths, scores, and labels are editable in the PPTX. The sequence uses the saved 0544 parent-group-7 scores and allocation; archive topology and motion timing are schematic.

**patent-v2.pdf** is the complete static version. **allocation-walkthrough.gif** is the allocation animation for reuse elsewhere. The PDF shows all animation builds at once.

**patent-v2.pptx** contains 28 static slides with editable titles, body text, footnotes, page numbers, and figure labels. The PDF contains 17 pages. Original pages 7, 9, and 10 expand into successive builds, retaining the same printed page number within each sequence. Advance manually through each build. The PPTX contains no GIFs or automatic animations. Equations, plot geometry, table rules, and the logo remain graphics. Text boxes use the original fonts and measured positions, with existing line breaks preserved.

The presentation emphasizes the technical solution and patent value. Published discovery systems appear on page 5. Design rationale is integrated with the distinguishing features. Two supporting pages summarize exploration behavior and one evaluated construction. Detectability, alternative implementations, and commercial applications each have their own page. Huawei branding and the supplied slide style are preserved. Display equations use native MathML typesetting with STIX Two Math.

The slides contain the technical explanation, equation definitions, and figure labels. Source details and comparison limitations are retained in speaker notes and **evidence.json**.

The Qwen3-8B ablation uses the saved measurement artifact, not a new experiment. It matches coder attempts, not total compute. The scientific result uses the exact saved 0544 construction and score; it is not a refreshed state-of-the-art or patentability assessment. The supplied inventor declarations require inventor confirmation before submission.

Detailed component comparisons and full-pipeline trajectories remain in the evidence record rather than in a separate experiment section. Historical report snapshots are transcribed in **report-experiments.json**, with source pages and their scope. The build retains the system trajectories from saved run metadata. Unequal-budget component comparisons are not presented as controlled causal ablations. Report snapshots are not silently mixed with later final run totals.

Construction figures use the same score-preserving orientation as `plot_erdos.py`. The embedded evidence retains each raw saved array alongside its display-oriented array. Scheduling colors distinguish the planner, coder, evaluation, allocation, and training roles.

For text or diagram edits, change **slides.template.html**, **deck.js**, or **framework.js**, then run `node output/patent_v2/build.mjs` from the repository root. This rebuilds the offline deck, framework player, and **evidence.json** provenance record. The build uses saved JSON only and does not import training code, execute generated programs, contact a server, or change run artifacts.

To rebuild the PPTX, run `render.mjs PRIVATE_BUILD_DIR --editable-pptx-assets` after the HTML build, then `export-pptx.mjs PRIVATE_BUILD_DIR`. Both scripts live in this directory. The renderer extracts visible text and positions from the HTML and captures the graphics with text removed. The exporter writes native PowerPoint text boxes and expands complete animation builds into static slides.

Add `--framework-motion` to the render command to export the framework video at 24 fps. The player respects reduced-motion preferences on opening. The PDF remains static.

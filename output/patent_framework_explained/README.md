# Patent presentation

Open **patent-v2.html** in a browser. It is self-contained, works offline, and needs no PowerPoint.

- Slides open fully visible so they can be read without a presenter. **Play** replays the animated builds.
- **Right / Space**: next slide (or next build while an animation is paused).
- **Left**: previous build or slide.
- **Page Down / Page Up**: change slide directly.
- **F**: full screen. **N**: speaker notes and sources. **O**: slide overview.
- **Play** animates the current pipeline, allocation, or scheduling slide.

**framework-animation.html** opens the framework explanation directly and plays offline. Use Play / Pause or Replay to control it. **framework-animation.mp4** is its 51-second video export, and **framework-animation.gif** is a looping preview. The six scenes define candidates, parent-child ancestry, the strategist, pilots, and follow-ups as they appear. The planning scene builds four calls in order: each receives the task, the same parent, and every earlier strategy for that parent. Inputs appear before the model produces each new strategy. Top-r / top-K retention, a frozen strategist, and one SPO² update with a fixed historical reference remain explicit. Scores display six decimals for readability; **evidence.json** retains the original precision. The sequence uses the saved 0544 parent-group-7 scores and allocation; archive topology, later unscored nodes, and motion timing are schematic.

The latest strategist-input animation is included in both HTML files and the video/GIF. The existing PPTX and PDF have not been regenerated for this animation revision. The PPTX retains its previous static planning diagram, with editable shapes, paths, scores, and labels.

**patent-v2.pdf** is the complete static version. **allocation-walkthrough.gif** is the allocation animation for reuse elsewhere. The PDF shows all animation builds at once.

**patent-v2.pptx** contains 30 static slides with editable titles, body text, footnotes, page numbers, and figure labels. The PDF contains 19 pages. Original pages 7, 9, and 10 expand into successive builds, retaining the same printed page number within each sequence. Advance manually through each build. The PPTX contains no GIFs or automatic animations. Equations, plot geometry, table rules, and the logo remain graphics. Text boxes use the original fonts and measured positions, with existing line breaks preserved.

The presentation includes the technical solution, an observed allocation, saved construction evolution, and the evaluated scientific result. Huawei branding and the supplied slide style are preserved. Display equations use native MathML typesetting with STIX Two Math.

The slides contain the technical explanation, equation definitions, and figure labels. Source details and comparison limitations are retained in speaker notes and **evidence.json**.

The Qwen3-8B ablation uses the saved measurement artifact, not a new experiment. It matches coder attempts, not total compute. The scientific result uses the exact saved 0544 construction and score; it is not a refreshed state-of-the-art or patentability assessment. The supplied inventor declarations require inventor confirmation before submission.

Three additional evidence pages connect experimental observations to design choices: coder learning and reward design, planning and coding models, and two runs of the full pipeline. Historical report snapshots are transcribed in **report-experiments.json**, with source pages and their scope. The build derives the system trajectories from saved run metadata. Unequal-budget component comparisons are not presented as controlled causal ablations. Report snapshots are not silently mixed with later final run totals.

Construction figures use the same score-preserving orientation as `plot_erdos.py`. The embedded evidence retains each raw saved array alongside its display-oriented array. Scheduling colors distinguish the planner, coder, evaluation, allocation, and training roles.

For text or diagram edits, change **slides.template.html**, **deck.js**, or **framework.js**, then run `node output/patent_framework_explained/build.mjs` from the repository root. This rebuilds the offline deck, framework player, and **evidence.json** provenance record. The build uses saved JSON only and does not import training code, execute generated programs, contact a server, or change run artifacts.

To rebuild the PPTX, run `render.mjs PRIVATE_BUILD_DIR --editable-pptx-assets` after the HTML build, then `export-pptx.mjs PRIVATE_BUILD_DIR`. Both scripts live in this directory. The renderer extracts visible text and positions from the HTML and captures the graphics with text removed. The exporter writes native PowerPoint text boxes and expands complete animation builds into static slides.

Add `--framework-motion` to the render command to export the framework video at 30 fps. Scene changes use a 0.25-second fade-out followed by a 0.35-second fade-in, without overlapping layouts. Motion completes early within each scene, leaving the completed diagram still for reading. The player respects reduced-motion preferences on opening. The PDF remains static.

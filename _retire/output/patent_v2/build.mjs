import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../..');
const run = 'runs/erdos_Qwen3.8-27B_1013-0544';
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const array = value => typeof value === 'string' ? JSON.parse(value) : value;
// Match plot_erdos.py's display convention without changing the saved construction.
function canonicalOrientation(h) {
  const edge = Math.max(1, Math.floor(h.length / 10));
  const mass = a => a.slice(0, edge).concat(a.slice(-edge)).reduce((s, v) => s + v, 0) / edge;
  const candidates = [h, h.map(v => 1 - v), [...h].reverse(), [...h].reverse().map(v => 1 - v)];
  return candidates.reduce((best, candidate) => mass(candidate) < mass(best) ? candidate : best);
}
const source = read('output/pdf/strategy_exploration_comparison.json');
const reportExperiments = read('output/patent_v2/report-experiments.json');
for (const key of ['coderLearning', 'directCoder', 'plannerCapacity', 'coderCapacity']) {
  assert(reportExperiments[key].every(r => r.programsApprox > 0 && r.positiveRewardPct >= 0 && r.positiveRewardPct <= 100 && r.bestScore > 0.38 && r.bestScore < 0.4));
}
const ablation = Object.entries(source.runs).map(([label, value]) => ({
  label,
  directory: value.directory,
  rows: value.steps.map(row => ({
    step: row.step, attempts: row.attempts, valid: row.valid,
    distinct: row.distinct_non_parent_per_100_attempts,
    copies: row.parent_copies_per_100_attempts,
  })),
}));
assert.equal(ablation.length, 2);
assert(ablation.every(x => x.rows.length === 20));
assert(ablation.every(x => x.rows.every(r => r.attempts === 192)));
const summary = read(`${run}/step00/step00.summary.json`);
const allocations = summary.strategy_pilot_allocation
  .filter(x => x.parent_group === 7)
  .sort((a, b) => a.strategy_index - b.strategy_index);
assert.deepEqual(allocations.map(x => x.pilot_count), [11, 11, 11, 11]);
assert.deepEqual(allocations.map(x => x.followup_count), [0, 0, 17, 3]);
const snapshots = [0, 3, 7, 10].map(step => {
  const folder = `${run}/step${String(step).padStart(2, '0')}`;
  const rows = fs.readdirSync(path.join(root, folder))
    .filter(name => /_rollout\d+\.meta\.json$/.test(name))
    .map(name => ({ source: `${folder}/${name}`, ...read(`${folder}/${name}`) }))
    .filter(row => row.valid && row.raw_score > 0 && row.construction)
    .sort((a, b) => a.raw_score - b.raw_score);
  const winner = rows[0];
  assert(winner, `Missing evaluated construction in step ${step}`);
  const h = array(winner.construction);
  assert(Array.isArray(h) && h.every(Number.isFinite));
  return { step, score: winner.raw_score, h, displayH: canonicalOrientation(h), n: h.length,
    source: winner.source, nodeId: winner.node_id };
});
const bestRecord = read(`${run}/step10/step10.best.json`);
const best = { score: bestRecord.raw_score, h: array(bestRecord.construction),
  step: bestRecord.found_step, nodeId: bestRecord.state_id,
  source: `${run}/step10/step10.best.json` };
best.displayH = canonicalOrientation(best.h);
assert.equal(best.h.length, 700);
assert(Math.abs(best.h.reduce((a, b) => a + b, 0) / best.h.length * 2 - 1) < 1e-9);
assert(best.h.every(x => x >= -1e-9 && x <= 1 + 1e-9));
function c5(h) {
  let peak = -Infinity;
  for (let lag = 1 - h.length; lag < h.length; lag++) {
    let overlap = 0;
    for (let i = Math.max(0, lag); i < Math.min(h.length, h.length + lag); i++) {
      overlap += h[i] * (1 - h[i - lag]);
    }
    peak = Math.max(peak, overlap * 2 / h.length);
  }
  return peak;
}
const peak = c5(best.h);
assert(Math.abs(peak - best.score) < 1e-10, 'Stored construction/score mismatch');
for (const record of [...snapshots, best]) {
  assert(Math.abs(c5(record.displayH) - record.score) < 1e-10, 'Display orientation changed the score');
}
const curve = Array.from({ length: 11 }, (_, step) => {
  const stem = `step${String(step).padStart(2, '0')}`;
  const s = read(`${run}/${stem}/${stem}.summary.json`);
  assert(s.completed);
  return { step, score: s.best_seen_raw_score };
});
const tree = read(`${run}/search_tree.json`);
const lineage = [];
let id = best.nodeId;
const visited = new Set();
while (id) {
  assert(!visited.has(id), 'Cyclic recorded lineage');
  visited.add(id);
  const node = tree.nodes[id];
  assert(node, `Missing ancestor ${id}`);
  lineage.unshift({ id, score: node.raw_problem_score });
  id = node.parent_id;
}
const systemComparison = [
  { label: 'Entropic-GRPO', directory: 'runs/erdos_Qwen3.8-27B_GRPO-Strategy1' },
  { label: 'SPO²', directory: run },
].map(record => {
  const config = read(`${record.directory}/config.json`);
  const rows = Array.from({ length: 7 }, (_, step) => {
    const stem = `step${String(step).padStart(2, '0')}`;
    const s = read(`${record.directory}/${stem}/${stem}.summary.json`);
    assert(s.completed);
    return { step, score: s.best_seen_raw_score,
      plannedPrograms: s.planned_rollouts ?? config.groups_per_step * config.group_size,
      source: `${record.directory}/${stem}/${stem}.summary.json` };
  });
  return { ...record, rows, programsPerStep: rows[0].plannedPrograms,
    followupsPerParent: config.group_size - config.strategies_per_parent * config.pilot_programs_per_strategy };
});
const allocationDecisions = [];
for (let step = 0; step <= 6; step++) {
  const stem = `step${String(step).padStart(2, '0')}`;
  const s = read(`${run}/${stem}/${stem}.summary.json`);
  for (const parentGroup of new Set(s.strategy_pilot_allocation.map(r => r.parent_group))) {
    const rows = s.strategy_pilot_allocation.filter(r => r.parent_group === parentGroup);
    assert.equal(rows.length, 4);
    const pilotBest = Math.min(...rows.map(r => r.pilot_best_raw_score).filter(Number.isFinite));
    const finalBest = Math.min(...rows.map(r => r.final_best_raw_score).filter(Number.isFinite));
    allocationDecisions.push({ step, parentGroup, pilotBest, finalBest, improved: finalBest < pilotBest });
  }
}
const followupSource = `${run}/step00/step00_group07_rollout050.meta.json`;
const followup = read(followupSource);
assert.equal(followup.strategy_rollout_phase, 'adaptive');
assert(lineage.some(n => n.id === followup.node_id));
assert(followup.raw_score > curve[0].score, 'The retained follow-up was not the step-0 global winner');
const followupEvidence = {
  source: followupSource, nodeId: followup.node_id, step0GlobalBest: curve[0].score,
  decisions: allocationDecisions, parentDecisions: allocationDecisions.length,
  improvedParents: allocationDecisions.filter(r => r.improved).length,
  ancestry: [0, 1, 3, 6, 10].map(step => ({ step, ...lineage[step + 1] })),
};
assert.equal(lineage.length, 12, 'The selected lineage rows require one recorded ancestor per completed step');
assert.equal(followupEvidence.parentDecisions, 56);
assert.equal(followupEvidence.improvedParents, 16);
const data = {
  title: 'Hierarchical Bandit-Guided Scientific Discovery',
  date: '2026-10-27', run,
  ablation, allocations, snapshots, best, curve, lineage,
  reportExperiments, systemComparison, followupEvidence,
  allocationPilotBest: Math.min(...allocations.map(x => x.pilot_best_raw_score).filter(Number.isFinite)),
  allocationFinalBest: Math.min(...allocations.map(x => x.final_best_raw_score).filter(Number.isFinite)),
  benchmarkReference: { value: 0.380868, source: 'Historical SimpleTES comparison supplied with the presentation. Not a current record survey.' },
  measurement: {
    source: 'output/pdf/strategy_exploration_comparison.json',
    comparison: source.construction_comparison,
    tolerance: source.construction_atol,
    denominator: source.denominator,
    computeMatched: source.compute_matched,
    sameParents: source.same_parent_contexts,
  },
  sources: [
    { label: 'Supplied presentation', path: '/Users/mohammad/Downloads/IMG_7154 Large.pdf' },
    { label: 'Reference presentation, visual inspiration only', path: '/Users/mohammad/Downloads/IMG_7166 Large.pdf' },
    { label: 'Historical component comparisons', path: reportExperiments.source, pages: reportExperiments.pages, scope: reportExperiments.scope },
    { label: 'Strategy-conditioning ablation', path: 'output/pdf/strategy_exploration_comparison.json' },
    { label: 'Observed allocation', path: `${run}/step00/step00.summary.json`, parentGroup: 7 },
    { label: 'Final stored construction', path: best.source },
    { label: 'Best-so-far scores', path: `${run}/step00..step10/*.summary.json` },
    { label: 'Recorded lineage', path: `${run}/search_tree.json` },
    { label: 'Earlier combined-system run', path: 'runs/erdos_Qwen3.8-27B_GRPO-Strategy1/step00..step06/*.summary.json' },
    { label: 'Adaptive follow-up on final-best ancestry', path: followupSource },
    { label: 'Original Huawei logo', url: 'https://www.huawei.com/-/media/hcomponent-header/1.0.1.20260908162100/component/img/huawei_logo.png' },
  ],
};
const logo = fs.readFileSync(path.join(out, 'huawei-logo.png')).toString('base64');
let html = fs.readFileSync(path.join(out, 'slides.template.html'), 'utf8');
html = html.replace('__DATA_JSON__', JSON.stringify(data).replaceAll('<', '\\u003c'));
html = html.replace('/*__DECK_SCRIPT__*/', fs.readFileSync(path.join(out, 'deck.js'), 'utf8'));
html = html.replaceAll('__LOGO_BASE64__', logo);
assert(!html.includes('__DATA_JSON__') && !html.includes('__LOGO_BASE64__'));
assert(!/\bV[12]\b/.test(html), 'Remove version labels from the final presentation');
const introductoryText = html.slice(html.indexOf('<section'), html.indexOf('<section class="slide animated"')).replace(/<[^>]+>/g, ' ');
assert(!/\b(?:coder|strategist|strateg(?:y|ies)|planner)\b/i.test(introductoryText), 'Introduce the specialist LLM roles only with the proposed hierarchy');
assert.equal((html.match(/<math\b/g) || []).length, 6, 'All six display equations must use native math typesetting');
fs.writeFileSync(path.join(out, 'patent-v2.html'), html);
fs.writeFileSync(path.join(out, 'evidence.json'), JSON.stringify(data, null, 2) + '\n');
console.log(`Built offline deck. Final construction n=${best.h.length}, saved score=${best.score}, recomputed=${peak}`);
console.log(`Snapshot resolutions: ${snapshots.map(x => `${x.step}:${x.n}`).join(', ')}. Lineage: ${lineage.length} nodes.`);

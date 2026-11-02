import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { extractEditableText, extractFrameworkGeometry } from './editable-text.mjs';

const out = path.dirname(fileURLToPath(import.meta.url));
const build = process.argv[2];
if (!build || !fs.existsSync(build)) throw new Error('Pass an existing private render directory');
const chrome = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'].find(p=>fs.existsSync(p));
if(!chrome)throw new Error('Install a Chromium browser to render the deck');
const processChrome = spawn(chrome, ['--headless=new', '--no-first-run',
  '--no-default-browser-check', `--user-data-dir=${build}/chrome-profile`,
  '--remote-debugging-port=0', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
let ws;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timeout = setTimeout(() => reject(new Error('Chrome startup timed out')), 20000);
    processChrome.on('error', reject);
    processChrome.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\r\n]+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
  let sequence = 0;
  const pending = new Map();
  const exceptions = [];
  ws.addEventListener('message', event => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) {
      const p = pending.get(m.id); pending.delete(m.id);
      clearTimeout(p.timeout);
      if (m.error) p.reject(new Error(JSON.stringify(m.error))); else p.resolve(m.result);
    }
    if (m.method === 'Runtime.exceptionThrown') exceptions.push(m.params.exceptionDetails);
  });
  function command(method, params = {}, sessionId) {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${method}`)); }, 30000);
      pending.set(id, { resolve, reject, timeout });
      ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
  const call = (method, params) => command(method, params, sessionId);
  const evaluate = async expression => {
    const r = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await call('Page.enable'); await call('Runtime.enable'); await call('Network.enable');
  await call('Network.setBlockedURLs', { urls: ['http://*', 'https://*'] });
  await call('Emulation.setDeviceMetricsOverride', { width: 1280, height: 782, deviceScaleFactor: 1, mobile: false });
  await call('Page.navigate', { url: pathToFileURL(path.join(out, 'patent-v2.html')).href });
  for (let i = 0; i < 50; i++) {
    if (await evaluate('Boolean(window.deck)')) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  if (!await evaluate('Boolean(window.deck)')) throw new Error('Deck did not initialize');
  await evaluate(`document.head.insertAdjacentHTML('beforeend','<style id="qa-style">*{transition:none!important;animation:none!important}</style>')`);
  const report = { slides: [], browserExceptions: exceptions };
  const count = await evaluate('deck.state.count');
  if (count !== 19) throw new Error(`Expected 19 slides; received ${count}`);
  await evaluate('deck.show(6)');
  if (!await evaluate('deck.state.build === deck.state.maxBuild')) throw new Error('Reading a slide must show its complete content');
  await evaluate('deck.show(9,99)');
  const arrowOverlaps = await evaluate(`(() => {
    const svg=document.querySelector('#schedule svg');
    const boxes=[...svg.querySelectorAll('[data-schedule-activity]')].map(e=>({name:e.dataset.scheduleActivity,b:e.getBBox()}));
    return [...svg.querySelectorAll('[data-schedule-dependency]')].flatMap(e=>{
      const x=Number(e.getAttribute('x1')),a=Number(e.getAttribute('y1')),b=Number(e.getAttribute('y2'));
      return boxes.filter(({b:r})=>x+3>r.x&&x-3<r.x+r.width&&Math.max(a,b)+.7>r.y&&Math.min(a,b)-.7<r.y+r.height).map(({name})=>name);
    });
  })()`);
  if (arrowOverlaps.length) throw new Error('Schedule arrow overlaps activity: '+arrowOverlaps.join(', '));
  for (let i = 0; i < count; i++) {
    await evaluate(`deck.show(${i},99)`);
    await new Promise(resolve => setTimeout(resolve, 70));
    const measurement = await evaluate(`(() => {
      const slide=document.querySelector('.slide.active');
      const elements=[...slide.querySelectorAll('h1,h2,p,li,td,th,.takeaway,.score-large')].filter(e=>e.getBoundingClientRect().height&& !e.closest('.source-notes'));
      const overflow=elements.filter(e=>{const r=e.getBoundingClientRect();const host=e.closest('.content');return r.left < -1||r.right>1281||r.top< -1||r.bottom>670||(host&&r.bottom>host.getBoundingClientRect().bottom+1)||e.scrollWidth>e.clientWidth+3;}).map(e=>({text:e.textContent.slice(0,90),rect:[e.offsetLeft,e.offsetTop,e.offsetWidth,e.offsetHeight],bottom:e.getBoundingClientRect().bottom}));
      const svgs=[...slide.querySelectorAll('svg')];const svgOut=[];
      for(const s of svgs){const a=s.getBoundingClientRect();for(const t of s.querySelectorAll('text')){const r=t.getBoundingClientRect();if(r.left<a.left-3||r.right>a.right+3||r.top<a.top-3||r.bottom>a.bottom+3)svgOut.push(t.textContent);}}
      return {title:slide.dataset.title,overflow,svgOut};
    })()`);
    report.slides.push({ number: i + 1, ...measurement });
    const shot = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false, clip: { x: 0, y: 0, width: 1280, height: 720, scale: 1 } });
    fs.writeFileSync(path.join(build, `slide-${String(i + 1).padStart(2, '0')}.png`), Buffer.from(shot.data, 'base64'));
  }
  await evaluate('deck.show(8,3)');
  const pdf = await call('Page.printToPDF', { printBackground: true, displayHeaderFooter: false,
    preferCSSPageSize: true, paperWidth: 13.333333, paperHeight: 7.5,
    marginTop: 0, marginRight: 0, marginBottom: 0, marginLeft: 0 });
  fs.writeFileSync(path.join(out, 'patent-v2.pdf'), Buffer.from(pdf.data, 'base64'));
  await evaluate('deck.show(8,0)');
  const frameSpec = [{ stage: 0, delay: 1.8 }, { stage: 1, delay: 1.8 }];
  for (let j = 1; j <= 10; j++) frameSpec.push({ stage: 2, progress: j / 10, delay: .09 });
  frameSpec.push({ stage: 2, delay: 1.6 }, { stage: 3, delay: 3.1 });
  for (let i = 0; i < frameSpec.length; i++) {
    const f = frameSpec[i];
    await evaluate(`deck.show(8,${f.stage})`);
    if (f.progress) await evaluate(`document.querySelectorAll('.slide.active .allocation-bar').forEach(x=>x.setAttribute('width',Number(x.dataset.targetWidth)*${f.progress}))`);
    const shot = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false, clip: { x: 0, y: 0, width: 1280, height: 720, scale: 1 } });
    fs.writeFileSync(path.join(build, `gif-${String(i).padStart(2, '0')}.png`), Buffer.from(shot.data, 'base64'));
  }
  fs.writeFileSync(path.join(build, 'gif-frames.json'), JSON.stringify(frameSpec, null, 2));
  if(process.argv.includes('--framework-motion')){
    await evaluate(`(() => {
      const forest=document.querySelector('[data-scene=forest]');
      if(forest.querySelectorAll('path').length!==32||forest.querySelectorAll('path[stroke="#C7000B"]').length!==2||forest.querySelectorAll('circle[fill="#C7000B"]').length!==2)throw new Error('Ancestry must recolor existing links and nodes');
      if(forest.querySelector('circle[fill=none]'))throw new Error('Unwanted archive ring');
      const nodes=[...forest.querySelectorAll('circle')].map(e=>[Number(e.getAttribute('cx')),Number(e.getAttribute('cy'))]);nodes.push([555,265]);
      for(const edge of forest.querySelectorAll('path')){
        const ends=[edge.getPointAtLength(0),edge.getPointAtLength(edge.getTotalLength())];
        if(!ends.every(p=>nodes.some(n=>Math.hypot(p.x-n[0],p.y-n[1])<.01)))throw new Error('Archive edge misses node center');
      }
      if(document.querySelectorAll('[data-coder]').length!==4)throw new Error('Expected four shared coder calls');
      for(const p of [.2,.3,.4,.7,1]){
        framework.setStage(2,p,true);
        if([...document.querySelectorAll('[data-pilot]')].some(e=>Number(e.style.opacity)>0)&&[...document.querySelectorAll('[data-coder],[data-coder-output]')].some(e=>Number(e.style.opacity)<1))throw new Error('Programs before coder output');
      }
      for(const p of [0,.3,.6,.7,.9,1]){
        framework.setStage(5,p,true);
        if([...document.querySelectorAll('[data-new-node]')].some(e=>Number(e.style.opacity)>0)&&Number(document.querySelector('[data-next-input]').getAttribute('stroke-dashoffset'))!==0)throw new Error('Generation before coder input');
      }
      const calls=[...document.querySelectorAll('[data-strategy-call]')];
      if(calls.length!==4)throw new Error('Missing strategist calls');
      if(document.querySelectorAll('[data-strategist-model]').length!==1)throw new Error('Expected one frozen strategist');
      calls.forEach((g,i)=>{
        const ids=[...g.querySelectorAll('[data-prior-strategy]')].map(e=>Number(e.dataset.priorStrategy));
        if(JSON.stringify(ids)!==JSON.stringify(Array.from({length:i},(_,j)=>j)))throw new Error('Incomplete strategy history');
        if(!g.querySelector('[data-call-inputs]').textContent.includes('Task + parent'))throw new Error('Missing parent input');
      });
      for(let p=0;p<=1;p+=.01){
        framework.setStage(1,p,true);
        const active=Math.min(3,Math.floor(p*4));
        calls.forEach((g,i)=>{
          if(Number(g.querySelector('[data-call-inputs]').style.opacity)!==(i===active?1:0))throw new Error('Wrong context displayed');
          if(i>active&&Number(g.querySelector('[data-call-result]').style.opacity)>0)throw new Error('Future strategy generated early');
          if(i<active&&Number(g.querySelector('[data-call-result] text').style.opacity)!==1)throw new Error('Earlier strategy not ready');
          if(Number(g.querySelector('[data-call-in-head]').style.opacity)>0&&Number(g.querySelector('[data-call-in]').getAttribute('stroke-dashoffset'))>1e-9)throw new Error('Strategist arrowhead appears before its line');
        });
        document.querySelectorAll('[data-history-transfer]').forEach(g=>{if(Number(g.style.opacity)>0&&Number(calls[Number(g.dataset.historyTransfer)].querySelector('[data-call-result] text').style.opacity)!==1)throw new Error('History copy before strategy is complete');});
      }
      framework.setStage(3);
      const extras=[...document.querySelectorAll('[data-extra]')];
      const extraTargets=extras.map(e=>[Number(e.getAttribute('cx')),Number(e.getAttribute('cy'))]);
      if(extras.length!==20)throw new Error('Allocation must preserve all 20 follow-ups');
      for(let p=0;p<=1;p+=.01){
        framework.setStage(3,p,true);
        const points=extras.map(e=>[Number(e.getAttribute('cx')),Number(e.getAttribute('cy'))]);
        points.forEach(([x,y],i)=>{
          const first=i<17?0:17;
          if(Math.abs((x-points[first][0])-(extraTargets[i][0]-extraTargets[first][0]))>.001||y!==extraTargets[i][1])throw new Error('Follow-up allocation loses its formation');
          if(points.slice(i+1).some(([xx,yy])=>Math.hypot(x-xx,y-yy)<12))throw new Error('Follow-up dots overlap while moving');
        });
      }
      framework.setStage(2);
      const pilotDots=[...document.querySelectorAll('[data-pilot]')];
      const pilotTargets=pilotDots.map(e=>[e.getAttribute('cx'),e.getAttribute('cy')]);
      if(pilotDots.length!==44)throw new Error('Expected 11 trial programs per strategy');
      for(let p=0;p<=1;p+=.005){
        framework.setStage(2,p,true);
        const alpha=Number(pilotDots[0].style.opacity);
        pilotDots.forEach((e,i)=>{
          if(e.getAttribute('cx')!==pilotTargets[i][0]||e.getAttribute('cy')!==pilotTargets[i][1]||Number(e.style.opacity)!==alpha)throw new Error('Trial dots must reveal together in their final positions');
        });
        if([...document.querySelectorAll('[data-pilot-count]')].some(e=>Number(e.style.opacity)>0)&&alpha<1)throw new Error('Trial count appears before the complete row');
      }
      for(const stage of [0,1]){
        for(const q of [0,.25,.5,.75,.99999]){
          framework.seek(framework.starts[stage+1]-.55+q*1.4);
          const anchors=document.querySelector('.framework-transition').children[stage].lastElementChild;
          const dot=stage?anchors.lastElementChild:anchors.firstElementChild.querySelector('circle');
          if(stage===0&&Number(anchors.firstElementChild.style.opacity)!==1)throw new Error('Parent disappears entering prompt');
          if(q>.99){const point=dot.getBoundingClientRect();framework.setStage(stage+1,0);const next=document.querySelector(stage?'[data-scene=parent] circle':'[data-strategy-call="0"] [data-planning-parent]').getBoundingClientRect();if(Math.abs(point.x-next.x)>.01||Math.abs(point.y-next.y)>.01||Math.abs(point.width-next.width)>.01)throw new Error('Parent jumps between scenes');}
        }
      }
      for(let i=0;i<3;i++){
        framework.setStage(1,(i+.999999)/4);
        const copy=document.querySelector('[data-history-transfer="'+i+'"]'),node=copy.querySelector('circle'),label=copy.querySelector('text');
        if(Math.abs(Number(node.getAttribute('cx'))-54)>.001||Math.abs(Number(node.getAttribute('cy'))-(205+i*58))>.001||Math.abs(Number(label.getAttribute('x'))-72)>.001)throw new Error('Strategy does not settle into next prompt');
        if(!document.querySelector('[data-strategy-call="'+i+'"] [data-prompt-title]').textContent.includes('Next prompt'))throw new Error('Missing prompt preparation label');
      }
      framework.setStage(4);
      const modelBox=document.querySelector('[data-coder-body]').getBoundingClientRect();
      framework.setStage(5,0);
      const nextBox=document.querySelector('[data-next-model]').getBoundingClientRect();
      if(['x','y','width','height'].some(k=>Math.abs(modelBox[k]-nextBox[k])>.01))throw new Error('Updated coder jumps entering next step');
      for(let p=0;p<=1;p+=.01){
        framework.setStage(5,p,true);
        const parent=document.querySelector('[data-scene=parent]').transform.baseVal.consolidate().matrix;
        const input=document.querySelector('[data-next-input]');
        if(Number(document.querySelector('[data-next-input-head]').style.opacity)>0&&Number(input.getAttribute('stroke-dashoffset'))>1e-9)throw new Error('Coder arrowhead appears before its line');
        if(Number(input.style.opacity)>0&&(Math.abs(parent.e-480)>.001||Math.abs(parent.f-280)>.001))throw new Error('Coder input before child becomes parent');
        if([...document.querySelectorAll('[data-new-node]')].some(e=>Number(e.style.opacity)>0)&&Number(input.getAttribute('stroke-dashoffset'))!==0)throw new Error('Programs appear before coder receives inputs');
      }
      framework.setStage(5);
      if(document.querySelector('[data-next-output],[data-next-output-label]'))throw new Error('Dangling generation line remains');
      if(document.querySelector('[data-next-parent-label]').textContent!=='Next parent')throw new Error('Saved child never becomes the parent');
      if([...document.querySelectorAll('[data-forest-tree="3"]')].some(e=>Number(e.style.opacity)!==0))throw new Error('Rightmost archive tree overlaps coder');
      const coderLeft=document.querySelector('[data-next-model]').getBoundingClientRect().left;
      if([...document.querySelectorAll('[data-forest-tree]')].filter(e=>Number(e.style.opacity)>0).some(e=>e.getBoundingClientRect().right>coderLeft-20))throw new Error('Archive and coder need a clear gutter');
      document.querySelectorAll('[data-new-node]').forEach((e,i)=>{if(Number(e.getAttribute('cx'))!==410+i*70||Number(e.getAttribute('cy'))!==415)throw new Error('Generated program misses its tree position');});
      framework.setStage(3);
      const attributes=e=>['cx','cy','r','fill','stroke'].map(k=>e.getAttribute(k));
      const sourceRollouts=[...document.querySelectorAll('[data-pilot],[data-extra]')].map(attributes);
      framework.seek(framework.starts[4]-1.2);
      const batch=document.querySelector('.framework-transition').children[3].lastElementChild;
      if(batch.children.length!==64)throw new Error('Rollout batch lost programs');
      if(JSON.stringify([...batch.children].map(attributes))!==JSON.stringify(sourceRollouts))throw new Error('Rollout handoff changes its source dots');
      for(const q of [0,.3,.65,.74,.99]){
        framework.seek(framework.starts[4]-1.2+q*2.8);
        if([...batch.children].some(e=>Number(e.style.opacity)!==1))throw new Error('Rollouts disappear while gathering');
        if(q<=.65&&Number(batch.parentElement.children[1].style.opacity)!==0)throw new Error('Training scene arrives before batch gathers');
      }
      framework.seek(framework.starts[4]+1.6-.00001);
      const gathered=[...batch.children].map(attributes);
      framework.seek(framework.starts[4]+1.6);
      if(JSON.stringify([...document.querySelectorAll('[data-outcome]')].map(attributes))!==JSON.stringify(gathered))throw new Error('Batch jumps entering training');
      for(let t=36.6;t<=58;t+=.1){
        framework.seek(t);
        if(JSON.stringify([...document.querySelectorAll('[data-outcome]')].map(attributes))!==JSON.stringify(gathered))throw new Error('Rollout grid moves again after gathering');
      }
      for(let p=0;p<=1;p+=.01){
        framework.setStage(4,p,true);
        if(document.querySelector('[data-adapter-status]').textContent==='Updated'&&[...document.querySelectorAll('[data-outcome]')].some(e=>Number(e.style.opacity)>0))throw new Error('LoRA updates before every rollout contributes');
      }
      const outcomeDots=[...document.querySelectorAll('[data-outcome]')];
      if(document.querySelector('[data-theory-batch],[data-learning-archive]'))throw new Error('Redundant training layout remains');
      for(const t of [40,45,...framework.trainingTimes,53,56]){
        framework.seek(t);
        if(outcomeDots.length!==64||outcomeDots.some(e=>Number(e.style.opacity)!==1))throw new Error('Training drops a rollout');
      }
      framework.seek(framework.trainingTimes[0]);
      const reference=Number(document.querySelector('[data-history-reference]').dataset.referenceValue);
      if(Math.abs(reference-2.4420042144696796)>1e-12)throw new Error('Historical moving reference changed');
      if(document.querySelector('[data-past-batch],[data-history-packet]'))throw new Error('Grouped history blocks remain');
      if(document.querySelector('.framework-figure').textContent.toLowerCase().includes('batch')||framework.phases.some(p=>p.join(' ').toLowerCase().includes('batch')))throw new Error('Unrequested batch terminology remains');
      const past=[...document.querySelectorAll('[data-history-rollout]')];
      if(past.length!==32)throw new Error('Missing individual historical rollouts');
      const expectedMeans=[1.3,5.8,6.2,3];
      for(let i=0;i<4;i++){
        const mean=past.slice(i*8,(i+1)*8).reduce((sum,e)=>sum+Math.exp(2*Number(e.dataset.reward)),0)/8;
        if(Math.abs(mean-expectedMeans[i])>1e-12)throw new Error('History dots do not reproduce the shared reference');
      }
      let beforeReference=1.3;
      for(const t of [39.7,40.2,40.75,41.6,42.15,43,43.55]){
        framework.seek(t);
        const ref=document.querySelector('[data-history-reference]'),value=Number(ref.dataset.referenceValue),edge=ref.querySelector('path');
        if(Math.abs(edge.getTotalLength()-182)>.001)throw new Error('The full reference line must move together');
        const point=edge.getPointAtLength(0).matrixTransform(ref.transform.baseVal.consolidate().matrix);
        if(Math.abs(point.x-(110+200*Math.log(value)))>.001)throw new Error('Reference line position does not match the moving average');
        if([...document.querySelectorAll('[data-spo-rollout],[data-grpo-rollout]')].some(e=>Number(e.style.opacity)>0))throw new Error('Current rollouts appear before the reference is fixed');
        if(value<beforeReference)throw new Error('Illustrated history update moved in the wrong direction');
        beforeReference=value;
      }
      if(Math.abs(beforeReference-reference)>1e-12)throw new Error('Historical reference did not reach its final value');
      framework.seek(framework.trainingTimes[0]);
      if(document.querySelector('[data-reference-status]').textContent!=='Fixed for this step')throw new Error('Missing fixed-reference explanation');
      const historyRight=Math.max(...past.map(e=>Number(e.getAttribute('cx'))+Number(e.getAttribute('r'))));
      const currentLeft=Math.min(...outcomeDots.map(e=>Number(e.getAttribute('cx'))-Number(e.getAttribute('r'))));
      if(currentLeft-historyRight<50)throw new Error('Past and current rollout clouds are not visually separated');
      const spo=[...document.querySelectorAll('[data-spo-rollout]')],grpo=[...document.querySelectorAll('[data-grpo-rollout]')];
      if(spo.length!==8||grpo.length!==8)throw new Error('Comparison must show the same eight rewards');
      spo.forEach((e,i)=>{
        const r=Number(e.dataset.reward),other=grpo[i],mean=i<4?.25:.65;
        if(r!==Number(other.dataset.reward))throw new Error('Comparison changes the rewards between methods');
        if(Math.abs(Number(e.dataset.advantage)-(Math.exp(2*r)/reference-1)/2)>1e-12)throw new Error('SPO weight differs from the tracker');
        if(Math.abs(Number(other.dataset.advantage)-(r-mean)/Math.sqrt(.0125))>1e-12)throw new Error('GRPO group normalization is incorrect');
        const dot=e.querySelector('circle'),otherDot=other.querySelector('circle');
        if(Number(otherDot.getAttribute('cx'))-Number(dot.getAttribute('cx'))!==600||dot.getAttribute('cy')!==otherDot.getAttribute('cy'))throw new Error('Reward axis differs between methods');
      });
      if(!document.querySelector('[data-training-legend]').textContent.includes('Direction only'))throw new Error('Sign-only comparison must be labeled');
      const shown=e=>{for(let p=e;p&&p.tagName!=='svg';p=p.parentElement){const s=getComputedStyle(p);if(s.visibility==='hidden'||Number(s.opacity)===0)return false;}return true;};
      for(let t=36.6;t<56.4;t+=.2){framework.seek(t);if(shown(document.querySelector('[data-adapter-box]')))throw new Error('Model appears before the learning explanation');}
      for(const t of [44,48.5,54.5,60,63]){
        framework.seek(t);
        if(Number(document.querySelector('[data-history-reference]').dataset.referenceValue)!==reference)throw new Error('Shared baseline changes during current-rollout comparison');
        const edge=document.querySelector('[data-baseline-line]'),start=edge.getPointAtLength(0);
        if(Math.abs(start.x-(110+200*Math.log(reference)))>.001||start.y!==183)throw new Error('Baseline is disconnected from the history reference');
      }
      framework.seek(54.5);
      spo.forEach(e=>{
        const a=Number(e.dataset.advantage),tip=Number(e.querySelector('[data-learning-arrow]').getAttribute('d').split(' ')[4]),y=Number(e.querySelector('circle').getAttribute('cy'));
        if(Math.abs(Math.abs(tip-(y-Math.sign(a)*7))-70*Math.abs(a)/Number(spo.at(-1).dataset.advantage))>1e-9)throw new Error('SPO arrow length must match its weight');
        const head=Number(e.querySelector('[data-learning-arrow]').getAttribute('d').split(' ')[7]);
        if(Math.sign(a)*(y-Math.sign(a)*7-head)<0)throw new Error('Short advantage arrowhead overlaps its dot');
      });
      let modelAppearances=0,wasVisible=false;
      for(let t=36.6;t<=64.4;t+=.05){
        framework.seek(t);const isVisible=shown(document.querySelector('[data-adapter-box]'));
        if(isVisible&&!wasVisible)modelAppearances++;wasVisible=isVisible;
        if(isVisible&&document.querySelector('[data-adapter-status]').textContent==='Updated'&&outcomeDots.some(e=>Number(e.style.opacity)>0))throw new Error('Update occurs before every rollout signal arrives');
      }
      if(modelAppearances!==1)throw new Error('Model must appear once in the continuous learning sequence');
      framework.seek(63);if(document.querySelector('[data-adapter-status]').textContent!=='Updated')throw new Error('Missing final LoRA update');
      if(!document.querySelector('[data-coder-body]').textContent.includes('Frozen'))throw new Error('Base weights must remain frozen');
      if(outcomeDots.some((e,i)=>Math.abs(Number(e.getAttribute('cx'))-800)>.001||Math.abs(Number(e.getAttribute('cy'))-(270+Math.floor(i/16)*11))>.001))throw new Error('Training signals miss the LoRA adapter');
      for (const t of framework.starts.slice(1)) {
        framework.seek(t);
        if (Number(document.querySelector('.framework-transition').style.opacity) !== 1) throw new Error('Missing scene handoff at '+t);
        if (Number(document.querySelector('.framework-layout').style.opacity) !== 0) throw new Error('Original scene overlays its handoff');
      }
      for (let t=0;t<framework.total;t+=.25) {
        framework.seek(t);
        const on=id=>Number(document.querySelector('[data-scene='+id+']').style.opacity)>0;
        if(on('learning')&&(on('plans')||on('pilots')||on('forest')))throw new Error('Overlapping scene layouts at '+t);
      }
      framework.setStage(5);
    })()`);
    const fps=30,seconds=await evaluate('deck.frameworkDuration');
    const encoder=spawn('/opt/homebrew/bin/ffmpeg',['-y','-loglevel','error','-f','image2pipe','-framerate',String(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',path.join(out,'framework-animation.mp4')],{stdio:['pipe','ignore','pipe']});
    let encoderError='';encoder.stderr.on('data',chunk=>{encoderError+=chunk;});
    const encoded=new Promise((resolve,reject)=>{encoder.on('error',reject);encoder.on('close',code=>code===0?resolve():reject(new Error('Video export failed: '+encoderError)));});
    encoded.catch(()=>{});
    for(let frame=0;frame<Math.ceil(seconds*fps);frame++){
      await evaluate(`deck.seekFramework(${frame/fps})`);
      const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false,clip:{x:0,y:0,width:1280,height:720,scale:1}});
      if(!encoder.stdin.write(Buffer.from(shot.data,'base64')))await once(encoder.stdin,'drain');
    }
    encoder.stdin.end();await encoded;
    for(let stage=0;stage<6;stage++){
      await evaluate(`deck.show(6,${stage})`);
      const shot=await call('Page.captureScreenshot',{format:'png',clip:{x:0,y:0,width:1280,height:720,scale:1}});
      fs.writeFileSync(path.join(build,`framework-stage-${stage+1}.png`),Buffer.from(shot.data,'base64'));
    }
    const trainingTimes=await evaluate('framework.trainingTimes');
    for(const [i,t] of trainingTimes.entries()){
      await evaluate(`deck.seekFramework(${t})`);
      const shot=await call('Page.captureScreenshot',{format:'png',clip:{x:0,y:0,width:1280,height:720,scale:1}});
      fs.writeFileSync(path.join(build,`framework-training-${i+1}.png`),Buffer.from(shot.data,'base64'));
    }
    await evaluate('deck.show(6,5)');
    const overlaps=await evaluate(`(()=>{const visible=e=>{for(let a=e;a&&a.tagName!=='svg';a=a.parentElement){const s=getComputedStyle(a);if(s.visibility==='hidden'||Number(s.opacity)===0)return false;}return true;};const texts=[...document.querySelectorAll('.framework-figure text')].filter(e=>visible(e)&&e.textContent.trim());return texts.flatMap((a,i)=>texts.slice(i+1).filter(b=>{const x=a.getBoundingClientRect(),y=b.getBoundingClientRect();return Math.min(x.right,y.right)-Math.max(x.left,y.left)>1&&Math.min(x.bottom,y.bottom)-Math.max(x.top,y.top)>1;}).map(b=>[a.textContent,b.textContent]));})()`);
    if(overlaps.length)throw new Error('Framework labels overlap: '+JSON.stringify(overlaps));
    console.log(`Exported ${seconds}-second framework animation at ${fps} fps.`);
  }
  if (process.argv.includes('--editable-pptx-assets')) {
    await evaluate(`document.head.insertAdjacentHTML('beforeend','<style>.build-caption{display:none!important}</style>')`);
    const slides = [];
    for (let page = 0; page < count; page++) {
      await evaluate(`deck.show(${page},0)`);
      const max = await evaluate('deck.state.maxBuild');
      for (let stage = 0; stage <= max; stage++) {
        await evaluate(`deck.show(${page},${stage})`);
        if (page === 6) {
          const still = await call('Page.captureScreenshot', {format:'png',clip:{x:0,y:0,width:1280,height:720,scale:1}});
          fs.writeFileSync(path.join(build,`framework-stage-${stage+1}.png`),Buffer.from(still.data,'base64'));
        }
        const info = await evaluate(`(() => {const s=document.querySelector('.slide.active');return {title:s.dataset.title,notes:s.querySelector('.source-notes')?.textContent.trim()||''};})()`);
        const texts = await evaluate(`(${extractEditableText.toString()})()`);
        const geometry = await evaluate(`(${extractFrameworkGeometry.toString()})()`);
        if (!texts.length) throw new Error(`No editable text found on page ${page+1}`);
        await evaluate(`document.head.insertAdjacentHTML('beforeend','<style id="text-mask">.slide *{color:transparent!important;-webkit-text-fill-color:transparent!important;text-shadow:none!important}.slide svg text{opacity:0!important}.slide math,.slide math *{color:#202328!important;-webkit-text-fill-color:#202328!important}.slide li::marker{color:#202328!important;-webkit-text-fill-color:#202328!important}</style>')`);
        if(geometry.length)await evaluate(`document.querySelector('.framework-figure').style.opacity='0'`);
        const shot = await call('Page.captureScreenshot', {format:'png',captureBeyondViewport:false,clip:{x:0,y:0,width:1280,height:720,scale:2}});
        const background = `editable-background-${page+1}-${stage}.png`;
        fs.writeFileSync(path.join(build,background),Buffer.from(shot.data,'base64'));
        await evaluate(`document.getElementById('text-mask').remove()`);
        if(geometry.length)await evaluate(`document.querySelector('.framework-figure').style.opacity='1'`);
        slides.push({page:page+1,stage,info,texts,geometry,background});
      }
    }
    fs.writeFileSync(path.join(build,'editable-pptx.json'),JSON.stringify({slides},null,2));
    console.log(`Captured editable text and text-free artwork for ${slides.length} static builds.`);
  }
  if (process.argv.includes('--pptx-assets')) {
    await evaluate(`document.head.insertAdjacentHTML('beforeend','<style>.build-caption{display:none!important}</style>')`);
    const slides = await evaluate(`[...document.querySelectorAll('.slide')].map(s=>({title:s.dataset.title,notes:s.querySelector('.source-notes')?.textContent.trim()||''}))`);
    const diagrams = [];
    for (const [slide, id] of [[6, 'pipeline'], [8, 'allocation'], [9, 'schedule']]) {
      await evaluate(`deck.show(${slide},99)`);
      const bounds = await evaluate(`(() => {const r=document.getElementById('${id}').getBoundingClientRect();return {x:Math.floor(r.x),y:Math.floor(r.y)-12,width:Math.ceil(r.width),height:Math.ceil(r.height)+24};})()`);
      const max = await evaluate('deck.state.maxBuild');
      const frames = [{ stage: max, delay: 2 }, { stage: 0, delay: 1.5 }];
      for (let stage = 1; stage <= max; stage++) {
        if (id === 'allocation' && stage === 2) {
          for (let j = 1; j <= 10; j++) frames.push({stage, progress:j/10, delay:.09});
        }
        frames.push({stage, delay:stage === max ? 3 : 1.8});
      }
      for (let j = 0; j < frames.length; j++) {
        const f = frames[j];
        await evaluate(`deck.show(${slide},${f.stage})`);
        if (f.progress) await evaluate(`document.querySelectorAll('.slide.active .allocation-bar').forEach(x=>x.setAttribute('width',Number(x.dataset.targetWidth)*${f.progress}))`);
        const shot = await call('Page.captureScreenshot', {format:'png',captureBeyondViewport:false,clip:{...bounds,scale:2}});
        f.file = `diagram-${slide+1}-${String(j).padStart(2,'0')}.png`;
        fs.writeFileSync(path.join(build,f.file),Buffer.from(shot.data,'base64'));
      }
      diagrams.push({slide:slide+1,bounds,frames});
    }
    fs.writeFileSync(path.join(build,'pptx-assets.json'),JSON.stringify({slides,diagrams},null,2));
  }
  fs.writeFileSync(path.join(build, 'layout-report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  console.log(`Rendered ${count} slides, static PDF, and ${frameSpec.length} GIF frames.`);
  await command('Target.closeTarget', { targetId });
} finally {
  ws?.close(); processChrome.kill('SIGTERM');
}

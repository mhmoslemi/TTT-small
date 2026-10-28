'use strict';
const DATA = JSON.parse(document.getElementById('deck-data').textContent);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const c = {red:'#c7000b', grey:'#626b76', plan:'#466b8a', code:'#25817c', eval:'#a77526', alloc:'#806096', train:'#bd343a'};
const label = (x,y,s,size=20,extra='') => `<text x="${x}" y="${y}" font-size="${size}" ${extra}>${esc(s)}</text>`;
const line = (x1,y1,x2,y2,extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
const rect = (x,y,w,h,extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" ${extra}/>`;
function svg(w,h,body,title){return `<svg class="data-figure" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${body}</svg>`;}
const center = (x,y,s,size=20,extra='') => label(x,y,s,size,`text-anchor="middle" ${extra}`);
const fragment = (n,body) => `<g class="fragment" data-build="${n}">${body}</g>`;
const arrow = (x1,y1,x2,y2,id='arr') => line(x1,y1,x2,y2,`class="edge" marker-end="url(#${id})"`);
const marker = id => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8e98a5"/></marker></defs>`;
function node(x,y,w,h,title,subtitle='',kind=''){
  return rect(x,y,w,h,`rx="3" class="node ${kind}"`)
    +center(x+w/2,y+(subtitle?h/2-5:h/2+7),title,20,'class="heavy"')
    +(subtitle?center(x+w/2,y+h/2+19,subtitle,16):'');
}

function background(){
  let b=marker('bg-arrow');
  const nodes=[['Task + parent','Problem-specific context'],['LLM program','Executable source'],['Construction','Program output'],['Evaluated score','Task-specific evaluator']];
  nodes.forEach((n,i)=>{let x=i*290;b+=node(x,30,255,85,n[0],n[1]);if(i<3)b+=arrow(x+255,72,x+285,72,'bg-arrow');});
  document.getElementById('background-flow').innerHTML=svg(1156,145,b,'The task and parent lead to a generated program. Execution produces a construction, which the task evaluator checks and scores.');
}

function pipeline(){
  let b=marker('pipe-arrow');
  b+=node(8,185,156,76,'Archive','Scores + ancestry');
  b+=node(8,30,156,60,'Parent p','One of P parents');
  b+=arrow(86,185,86,96,'pipe-arrow');
  const xs=[245,475,705,935];
  let plans=center(680,7,'Each new plan sees earlier plans to encourage a different approach',19,`style="fill:${c.plan}"`);
  plans+=arrow(164,60,235,60,'pipe-arrow');
  xs.forEach((x,i)=>{plans+=node(x,30,180,60,`Strategy ${i+1}`,'Plan for parent p','plan');if(i<3)plans+=arrow(x+180,60,x+220,60,'pipe-arrow');});
  b+=fragment(1,plans);
  let pilots='';xs.forEach(x=>{pilots+=arrow(x+90,90,x+90,121,'pipe-arrow');pilots+=node(x,125,180,50,'H pilot programs','','code');});
  xs.forEach(x=>{pilots+=arrow(x+90,175,x+90,193,'pipe-arrow');});
  pilots+=node(245,197,870,34,'CPU evaluation streams results as programs arrive','','eval');
  b+=fragment(2,pilots);
  let alloc=arrow(680,231,680,250,'pipe-arrow');
  alloc+=node(245,254,870,54,'Parent-relative hurdle bandit','Use pilot outcomes to allocate B follow-up programs','alloc');
  b+=fragment(3,alloc);
  let follow='';xs.forEach((x,i)=>{follow+=arrow(x+90,308,x+90,336,'pipe-arrow');follow+=node(x,340,180,44,`a${['₁','₂','₃','₄'][i]} new programs`,'','code');});
  b+=fragment(4,follow);
  let result=`<path d="M 1025 384 V 408 H 86 V 267" class="edge" marker-end="url(#pipe-arrow)"/>`;
  xs.slice(0,3).forEach(x=>{result+=line(x+90,384,x+90,408,'class="edge"');});
  result+=center(680,436,'Evaluate follow-ups and retain top candidates per strategy, then per parent',17);
  result+=label(104,288,'Update',17,'class="minor"');
  b+=fragment(5,result);
  document.getElementById('pipeline').innerHTML=svg(1156,441,b,'One parent produces four sequential strategies. Each receives equal coder pilots. The hurdle bandit allocates additional programs. Evaluated candidates update the archive.');
}

function allocation(){
  let b=marker('alloc-arrow')+label(8,22,'Strategy',18,'class="minor"')+label(245,22,'Equal pilot budgets',18,'class="minor"')+center(630,22,'Expected-best allocation',18,'class="minor"')+label(815,22,'Allocated follow-ups',18,'class="minor"');
  DATA.allocations.forEach((r,i)=>{
    const y=45+i*55;
    b+=label(8,y+25,`Strategy ${i+1}`,22);
    b+=rect(245,y,154,32,`fill="${c.plan}"`)+label(414,y+25,'11',22);
    b+=fragment(1,arrow(480,y+16,775,y+16,'alloc-arrow'));
    b+=fragment(2,rect(815,y,r.followup_count*14,32,`class="allocation-bar" data-target-width="${r.followup_count*14}" fill="${c.red}" style="transition:width .65s"`)+label(815+r.followup_count*14+15,y+25,String(r.followup_count),22));
    b+=line(8,y+44,1140,y+44,'stroke="#e5e8ed"');
  });
  b+=label(245,287,'44 initial programs',19,'class="minor"');
  b+=fragment(2,label(815,287,'20 additional programs',19,'class="minor"'));
  b+=fragment(3,label(8,338,'Best C₅: pilots',20,'class="minor"')+label(245,338,DATA.allocationPilotBest.toFixed(9),27)+label(480,338,'→',25)+label(540,338,DATA.allocationFinalBest.toFixed(9),29,`style="fill:${c.red}" class="heavy"`)+label(825,338,'After follow-ups',19,'class="minor"'));
  document.getElementById('allocation').innerHTML=svg(1156,350,b,'Every strategy gets 11 pilot programs. After evaluating those pilots, the bandit assigns the remaining 20 programs as zero, zero, 17, and 3. The observed best C5 improves after the follow-ups.');
}

function schedule(){
  const start=172,unit=7.5,ys=[45,77,109,141,173];
  const lanes=['Strategist','Coder','CPU evaluation','Allocation','Trainer'];
  const fills={[c.plan]:'#BCD0E0',[c.code]:'#B5D4CE',[c.eval]:'#E8D29C',[c.alloc]:'#D3C2DE',[c.train]:'#E2B8BB'};
  function laneFrame(offset){let s='';lanes.forEach((name,i)=>{s+=label(0,ys[i]+offset+6,name,18,'class="minor"');s+=line(start,ys[i]+offset,start+124*unit,ys[i]+offset,'class="grid"');});return s;}
  function activity(a,z,lane,name,color,offset=0){const x=start+a*unit,y=ys[lane]+offset-10;return rect(x,y,(z-a)*unit,20,`data-schedule-activity="${esc(name)}" fill="${fills[color]}" stroke="${color}" stroke-width="0.8" rx="2"`)+center(x+(z-a)*unit/2,y+15,name,16,'style="fill:#25323B"');}
  function dependency(t,from,to,offset=0){const direction=Math.sign(to-from),x=start+t*unit;return line(x,ys[from]+offset+direction*12,x,ys[to]+offset-direction*13,'data-schedule-dependency="true" stroke="#687381" stroke-width="1.25" marker-end="url(#schedule-arrow)"');}
  const arrowhead='<defs><marker id="schedule-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="#687381"/></marker></defs>';
  let b=arrowhead+label(0,17,'(a) Sequential execution',20,'class="heavy"')+label(480,17,'Each stage waits for the previous stage',17,'class="minor"')+laneFrame(0);
  b+=activity(0,12,0,'Plans',c.plan)+activity(12,30,1,'Pilot programs',c.code)+activity(30,47,2,'Evaluate pilots',c.eval)+activity(47,57,3,'Allocate',c.alloc)+activity(57,78,1,'Follow-up programs',c.code)+activity(78,96,2,'Evaluate outputs',c.eval)+activity(96,116,4,'Scoring / backward',c.train)+activity(116,124,4,'Update',c.train);
  b+=dependency(12,0,1)+dependency(30,1,2)+dependency(47,2,3)+dependency(57,3,1)+dependency(78,1,2)+dependency(96,2,4);
  const offset=208;b+=label(0,226,'(b) Dependency-aware execution',20,'class="heavy"')+label(480,226,'Follow-ups start once a parent’s pilot results are complete',17,'class="minor"')+laneFrame(offset);
  b+=fragment(1,activity(0,12,0,'Plans',c.plan,offset)+activity(12,30,1,'Pilot programs',c.code,offset)+activity(18,40,2,'Evaluate pilots',c.eval,offset)+dependency(12,0,1,offset)+dependency(18,1,2,offset));
  b+=fragment(2,activity(30,49,3,'Allocate per parent',c.alloc,offset)+activity(39,51,1,'Parent a',c.code,offset)+activity(51,63,1,'Parent b',c.code,offset)+activity(48,78,2,'Evaluate follow-ups',c.eval,offset)+dependency(30,2,3,offset)+label(695,273,'Follow-ups start per ready parent',17,`style="fill:${c.code}"`));
  b+=fragment(3,activity(63,86,4,'Scoring / backward',c.train,offset)+activity(86,95,4,'Update',c.train,offset)+label(start+98*unit,ys[4]+offset+6,'Step complete',18,'class="heavy"')+label(725,350,'Training overlaps remaining evaluations',17,`style="fill:${c.train}"`));
  document.getElementById('schedule').innerHTML=svg(1156,406,b,'Sequential execution waits for each complete stage. Dependency-aware execution evaluates programs as they arrive and allocates follow-ups per ready parent. When generation releases the shared GPUs, training overlaps remaining CPU evaluations. One optimizer update waits for all required results and gradients.');
}

function chartAxes(w,h,domainX,domainY,xticks,yticks,formatter=(v)=>String(v),left=50){
  const m={l:left,r:17,t:16,b:48};
  const x=v=>m.l+(v-domainX[0])/(domainX[1]-domainX[0])*(w-m.l-m.r);
  const y=v=>h-m.b-(v-domainY[0])/(domainY[1]-domainY[0])*(h-m.t-m.b);
  let body='';
  yticks.forEach(v=>{body+=line(m.l,y(v),w-m.r,y(v),'class="grid"')+label(m.l-10,y(v)+5,formatter(v),15,'text-anchor="end" class="axis"');});
  xticks.forEach(v=>{body+=label(x(v),h-m.b+24,String(v),15,'text-anchor="middle" class="axis"');});
  body+=line(m.l,h-m.b,w-m.r,h-m.b,'stroke="#98a1ad"');
  return {x,y,body,m};
}
function ablationChart(field,id,max){
  const w=550,h=250;
  const a=chartAxes(w,h,[0,19],[0,max],[0,4,8,12,16,19],max===100?[0,25,50,75,100]:[0,10,20,30,40]);
  let b=a.body;
  DATA.ablation.forEach((run,index)=>{
    const stroke=index===0?c.red:c.grey;
    b+=`<polyline points="${run.rows.map(r=>`${a.x(r.step).toFixed(2)},${a.y(r[field]).toFixed(2)}`).join(' ')}" class="line" stroke="${stroke}" stroke-width="3.1"/>`;
    const last=run.rows.at(-1);const dy=field==='copies'?(index===0?-12:22):(index===0?-12:-12);
    b+=label(a.x(19)-3,a.y(last[field])+dy,`${last[field].toFixed(1)}%`,17,`text-anchor="end" style="fill:${stroke}"`);
  });
  b+=center((a.m.l+w-a.m.r)/2,h-4,'Training step',17,'class="axis"');
  document.getElementById(id).innerHTML=svg(w,h,b,field==='copies'?'Parent copies as a percent of coder attempts. Without strategist rises to 92.2 percent; with strategist ends at 2.1 percent.':'Distinct valid non-parent outputs per 100 attempts. With strategist ends at 14.1 percent and without at 1.0 percent.');
}

function construction(h,w,height,mini=false){
  const m={l:60,r:15,t:mini?10:17,b:mini?39:43};
  const x=v=>m.l+v/2*(w-m.l-m.r), y=v=>height-m.b-v*(height-m.t-m.b);
  let b='';[0,.5,1].forEach(v=>{b+=line(m.l,y(v),w-m.r,y(v),'class="grid"')+label(m.l-8,y(v)+4,String(v),mini?13:15,'text-anchor="end" class="axis"');});
  [0,1,2].forEach(v=>{b+=label(x(v),height-m.b+20,String(v),mini?13:15,'text-anchor="middle" class="axis"');});
  let d=`M ${x(0).toFixed(2)} ${y(h[0]).toFixed(2)}`;
  for(let i=0;i<h.length;i++){d+=` H ${x(2*(i+1)/h.length).toFixed(2)}`;if(i<h.length-1)d+=` V ${y(h[i+1]).toFixed(2)}`;}
  b+=`<path d="${d}" class="line" stroke="${c.red}" stroke-width="${mini?1.5:1.9}"/>`;
  b+=center((m.l+w-m.r)/2,height-3,'x',17,'class="axis"');
  const mid=(m.t+height-m.b)/2;
  b+=center(12,mid,'h(x)',16,`class="axis" transform="rotate(-90 12 ${mid})"`);
  return svg(w,height,b,`Saved step function with ${h.length} equal-width pieces on the interval zero to two. Values lie between zero and one.`);
}
function snapshots(){
  document.getElementById('snapshots').innerHTML=DATA.snapshots.map(r=>`<div class="snapshot"><h2>Step ${r.step}<span>n = ${r.n} &nbsp; C₅ = ${r.score.toFixed(9)}</span></h2><div class="chart">${construction(r.displayH,550,160,true)}</div></div>`).join('');
  document.getElementById('best-construction').innerHTML=construction(DATA.best.displayH,650,285);
  document.getElementById('best-score').textContent=`C₅ ≤ ${DATA.best.score.toFixed(12)}`;
}
function bestCurve(){
  const w=440,h=285;const scores=DATA.curve.map(r=>r.score).concat(DATA.benchmarkReference.value);
  const lo=Math.min(...scores),hi=Math.max(...scores),pad=(hi-lo)*.12;
  const ticks=Array.from({length:4},(_,i)=>lo+(hi-lo)*i/3);
  const a=chartAxes(w,h,[0,10],[lo-pad,hi+pad],[0,2,4,6,8,10],ticks,v=>v.toFixed(5),85);
  let b=a.body;
  const ref=DATA.benchmarkReference.value;
  b+=line(a.x(0),a.y(ref),a.x(10),a.y(ref),'stroke="#9fa9b5" stroke-width="1.1"');
  b+=label(a.x(10)-2,a.y(ref)-9,'SimpleTES reference: 0.380868',15,'text-anchor="end" class="minor"');
  b+=`<polyline points="${DATA.curve.map(r=>`${a.x(r.step).toFixed(2)},${a.y(r.score).toFixed(2)}`).join(' ')}" class="line" stroke="${c.red}" stroke-width="2.7"/>`;
  b+=`<circle cx="${a.x(10)}" cy="${a.y(DATA.best.score)}" r="4" fill="${c.red}"/>`;
  b+=center((a.m.l+w-a.m.r)/2,h-3,'Completed step',17,'class="axis"');
  document.getElementById('best-curve').innerHTML=svg(w,h,b,'The best-so-far evaluated C5 bound decreases across completed steps zero through ten. The grey line marks the historical comparison reference, 0.380868.');
}

function experimentEvidence(){
  for(const [id,key] of [['learning-results','coderLearning'],['direct-results','directCoder'],['planner-results','plannerCapacity'],['coder-results','coderCapacity']]){
    document.getElementById(id).innerHTML=DATA.reportExperiments[key].map(r=>`<tr><td>${esc(r.label)}</td><td>${(r.programsApprox/1000).toFixed(1)}k</td><td>${r.positiveRewardPct.toFixed(2)}%</td><td>${r.bestScore.toFixed(6)}</td></tr>`).join('');
  }
  const w=720,h=320,a=chartAxes(w,h,[0,6],[.380853,.380998],[0,1,2,3,4,5,6],[.38086,.38090,.38094,.38098],v=>v.toFixed(5),105);
  let b=a.body;
  DATA.systemComparison.forEach((run,i)=>{
    const stroke=i===0?c.grey:c.red;
    b+=`<polyline points="${run.rows.map(r=>`${a.x(r.step).toFixed(2)},${a.y(r.score).toFixed(2)}`).join(' ')}" class="line" stroke="${stroke}" stroke-width="2.7"/>`;
    run.rows.forEach(r=>{b+=`<circle cx="${a.x(r.step)}" cy="${a.y(r.score)}" r="3" fill="${stroke}"/>`;});
  });
  b+=center((a.m.l+w-a.m.r)/2,h-3,'Completed step',17,'class="axis"');
  b+=label(a.m.l,12,'Best-so-far C₅ (lower is better)',16,'class="axis"');
  document.getElementById('system-progression').innerHTML=svg(w,h,b,'Two saved best-so-far C5 trajectories over completed steps zero through six. The combined SPO squared run reaches a lower bound. Other run settings also differ.');
  document.getElementById('earlier-system-score').textContent=DATA.systemComparison[0].rows.at(-1).score.toFixed(9);
  document.getElementById('historical-system-score').textContent=DATA.systemComparison[1].rows.at(-1).score.toFixed(9);
}

background();pipeline();allocation();schedule();ablationChart('distinct','diversity',40);ablationChart('copies','copies',100);snapshots();bestCurve();experimentEvidence();
const slides=[...document.querySelectorAll('.slide')];
slides.forEach((slide,i)=>{
  const footer=document.createElement('footer');footer.className='footer';
  footer.innerHTML=`<span>Huawei Confidential</span><span>Secret Level:</span><span class="page">Page ${i+1}</span><img alt="Huawei" src="data:image/png;base64,__LOGO_BASE64__">`;
  slide.append(footer);
});
const captions={
  pipeline:['Select parents from the archive','Each new plan sees earlier plans for the same parent','The coder generates equal pilot batches, and programs enter evaluation','Pilot outcomes determine each parent’s follow-up allocation','The coder generates the allocated follow-up programs','Retain selected children in the archive. All evaluated programs train the same coder.'],
  allocation:['Generate 11 pilot programs for each strategy','Use pilot outcomes to estimate the benefit of further sampling','Allocate the remaining 20 programs across the four strategies','The best saved C₅ bound improved after these follow-ups (lower is better)'],
  schedule:['Each stage waits for the previous stage to finish','Evaluate completed pilots while generation continues','Allocate follow-ups once each parent’s pilot results are complete','Once generation ends, train on evaluated programs while CPU evaluations continue'],
};
let current=0,build=0,timer=null;
const maxBuild=i=>Number(slides[i].dataset.builds||0);
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';}
function fit(){
  const host=document.getElementById('stage');const scale=Math.min(host.clientWidth/1280,host.clientHeight/720);
  slides.forEach(s=>s.style.transform=`scale(${Math.max(.1,scale)})`);
}
function render(){
  slides.forEach((s,i)=>{s.classList.toggle('active',i===current);s.setAttribute('aria-hidden',String(i!==current));
    s.querySelectorAll('[data-build]').forEach(el=>el.classList.toggle('visible',i!==current||Number(el.dataset.build)<=build));
  });
  const slide=slides[current],max=maxBuild(current);
  slide.querySelectorAll('.allocation-bar').forEach(el=>el.setAttribute('width',build>=2?el.dataset.targetWidth:0));
  const caption=slide.querySelector('.build-caption');if(caption)caption.textContent=captions[slide.dataset.animation][build];
  document.getElementById('counter').textContent=`${current+1} / ${slides.length}`;
  document.getElementById('current-title').textContent=slide.dataset.title;
  document.getElementById('build-status').textContent=max?`Build ${build+1}/${max+1}`:'';
  document.getElementById('play').disabled=!max;
  document.getElementById('previous').disabled=current===0&&build===0;
  document.getElementById('next').disabled=current===slides.length-1&&build===max;
  document.getElementById('notes-panel').textContent=slide.querySelector('.source-notes')?.textContent.trim()||'';
  document.getElementById('announce').textContent=`Slide ${current+1} of ${slides.length}. ${slide.dataset.title}. ${max?`Build ${build+1} of ${max+1}.`:''}`;
  history.replaceState(null,'',`#slide=${current+1}&build=${build}`);
  fit();
}
function show(index,step){stop();current=Math.max(0,Math.min(slides.length-1,index));build=Math.max(0,Math.min(maxBuild(current),step??maxBuild(current)));render();}
function next(){stop();if(build<maxBuild(current)){build++;render();}else if(current<slides.length-1)show(current+1);}
function previous(){stop();if(build>0){build--;render();}else if(current>0)show(current-1,maxBuild(current-1));}
document.getElementById('next').addEventListener('click',next);
document.getElementById('previous').addEventListener('click',previous);
document.getElementById('play').addEventListener('click',()=>{
  if(timer){stop();return;}if(!maxBuild(current))return;
  if(reduced){build=maxBuild(current);render();return;}
  if(build===maxBuild(current))build=0;
  render();document.getElementById('play').textContent='Pause';
  timer=setInterval(()=>{if(build>=maxBuild(current)){stop();return;}build++;render();},1900);
});
const overview=document.getElementById('overview');
slides.forEach((s,i)=>{const button=document.createElement('button');button.type='button';button.innerHTML=`<b>SLIDE ${String(i+1).padStart(2,'0')}</b>${esc(s.dataset.title)}`;button.addEventListener('click',()=>{overview.hidden=true;show(i);});overview.append(button);});
function toggleOverview(){overview.hidden=!overview.hidden;stop();}
function toggleNotes(){const p=document.getElementById('notes-panel');p.hidden=!p.hidden;}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{document.getElementById('announce').textContent='Full screen is unavailable in this browser window.';}}
document.getElementById('overview-button').addEventListener('click',toggleOverview);
document.getElementById('notes-button').addEventListener('click',toggleNotes);
document.getElementById('fullscreen').addEventListener('click',fullscreen);
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const key=e.key.toLowerCase();
  if(key==='escape'){overview.hidden=true;document.getElementById('notes-panel').hidden=true;stop();return;}
  if(!overview.hidden)return;
  if(['arrowright',' ','arrowleft','pagedown','pageup','home','end'].includes(key))e.preventDefault();
  if(key==='arrowright'||key===' ')next();else if(key==='arrowleft')previous();
  else if(key==='pagedown')show(current+1);else if(key==='pageup')show(current-1);
  else if(key==='home')show(0);else if(key==='end')show(slides.length-1);
  else if(key==='f')fullscreen();else if(key==='n')toggleNotes();else if(key==='o')toggleOverview();
});
addEventListener('resize',fit);
addEventListener('beforeprint',()=>{stop();document.querySelectorAll('.fragment').forEach(x=>x.classList.add('visible'));document.querySelectorAll('.allocation-bar').forEach(x=>x.setAttribute('width',x.dataset.targetWidth));});
addEventListener('afterprint',render);
window.deck={show,setBuild:n=>show(current,n),next,previous,get state(){return {current,build,count:slides.length,maxBuild:maxBuild(current)};},get data(){return DATA;}};
const initial=new URLSearchParams(location.hash.slice(1));
show(Number(initial.get('slide')||1)-1,initial.has('build')?Number(initial.get('build')):undefined);

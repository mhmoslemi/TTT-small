'use strict';

function createFramework(host, data) {
  const colors = {ink:'#202328',muted:'#606873',line:'#DCE0E5',plan:'#466B8A',code:'#25817C',eval:'#A77526',alloc:'#806096',red:'#C7000B'};
  const phases = [
    ['Select parents from the archive', 'Each parent supplies its program, construction, and evaluated score.'],
    ['Generate distinct plans for each parent', 'The frozen strategist sees its earlier plans. Each coder receives one plan and its parent.'],
    ['Generate and evaluate equal pilot batches', 'The coder samples 11 programs per strategy. Grey dots mark invalid outputs. CPU evaluation returns raw scores.'],
    ['Allocate follow-ups from the pilot outcomes', 'The bandit assigns this parent’s 20 remaining programs: 0, 0, 17, and 3.'],
    ['Evaluate follow-ups and accumulate training gradients', 'After generation ends, SPO² training starts on ready results while CPU evaluation continues.'],
    ['Retain children and update the shared coder', 'Selected children enter the archive. One optimizer update prepares the coder for the next step.'],
  ];
  const durations = [3.5,4,4.5,4.5,5,5.5];
  const starts = durations.map((_,i) => durations.slice(0,i).reduce((a,b)=>a+b,0));
  const total = durations.reduce((a,b)=>a+b,0);
  const escape = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (x,y,value,size=17,color=colors.ink,anchor='start',weight=400) => `<text x="${x}" y="${y}" font-size="${size}" style="fill:${color}" text-anchor="${anchor}" font-weight="${weight}">${escape(value)}</text>`;
  const circle = (x,y,r,fill,stroke=fill,extra='') => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.3" ${extra}/>`;
  const path = (d,color=colors.line,width=1.5,extra='') => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const box = (x,y,w,h,color,fill) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}" stroke="${color}" stroke-width="1.2"/>`;
  const group = (phase,body) => `<g class="framework-fragment" data-framework-phase="${phase}">${body}</g>`;
  const flow = (phase,d,color) => path(d,color,1.4)+`<path class="framework-trace" data-flow-phase="${phase}" d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" pathLength="1" opacity="0"/>`;
  const score = (x,y,value,color=colors.eval) => text(x,y,Number(value).toFixed(9),16,color,'middle');
  const xs = [375,595,815,1035];
  let drawing = '';

  let archive = text(0,18,'Search archive',21,colors.ink,'start',600)+text(0,41,'Programs, scores, lineage',15,colors.muted);
  archive += path('M 29 84 L 76 124 L 116 82 M 76 124 L 118 174 M 76 124 L 168 118 M 29 84 L 37 181', '#C4CCD5');
  [[29,84],[76,124],[116,82],[118,174],[37,181]].forEach(([x,y])=>{archive+=circle(x,y,5.5,'#E4E9EF','#B1BCC8');});
  archive += circle(29,84,11,'none',colors.plan)+circle(118,174,11,'none',colors.plan);
  archive += circle(168,118,13,'#FFFFFF',colors.red,'class="framework-parent-ring"')+circle(168,118,6,colors.red);
  archive += text(168,96,'Parent p',18,colors.ink,'middle',600)+score(168,146,data.lineage[0].score,colors.muted);
  archive += `<g data-through-phase="4">${text(0,223,'P parents selected',18,colors.ink,'start',600)}${text(0,247,'One parent expanded here',15,colors.muted)}</g>`;
  drawing += group(0,archive);

  let plans = text(270,18,'Frozen strategist',20,colors.plan,'start',600)+text(1120,18,'Sequential plans for the same parent',15,colors.muted,'end');
  plans += flow(1,'M 182 118 H 239 V 71 H 302',colors.plan);
  xs.forEach((x,i)=>{
    plans += `<g data-framework-stagger="${i}">${box(x-73,49,146,44,colors.plan,'#EEF3F8')}${text(x,77,`Strategy ${i+1}`,19,colors.plan,'middle',600)}</g>`;
    if(i<3)plans+=flow(1,`M ${x+73} 71 H ${xs[i+1]-73}`,colors.plan);
  });
  drawing += group(1,plans);

  let pilots = text(270,123,'Shared coder',16,colors.code,'start',600);
  xs.forEach((x,i)=>{
    pilots += flow(2,`M ${x} 94 V 136`,colors.code);
    for(let j=0;j<11;j++){
      const row=j<6?0:1,k=j<6?j:j-6,px=x+(k-(row?2:2.5))*14,py=145+row*15;
      const valid=j<data.allocations[i].pilot_valid_count;
      pilots+=circle(px,py,4.5,valid?colors.code:'#E3E7EC',valid?colors.code:'#CED5DE',`data-pilot-valid="${valid}"`);
    }
    pilots+=text(x,186,'11 pilots',17,colors.code,'middle');
    pilots+=flow(2,`M ${x} 193 V 205`,colors.eval)+circle(x,210,4,colors.eval);
    pilots+=score(x,237,data.allocations[i].pilot_best_raw_score);
  });
  pilots+=text(0,291,'CPU evaluation',18,colors.eval,'start',600)+text(0,315,'Raw C₅, lower is better',15,colors.muted);
  pilots+=text(270,216,'Best',13,colors.eval)+text(270,234,'pilot C₅',13,colors.eval);
  drawing += group(2,pilots);

  let allocation = text(270,267,'Parent-relative bandit',18,colors.alloc,'start',600)+text(1120,267,'20 follow-up programs',16,colors.alloc,'end');
  xs.forEach((x,i)=>{
    const n=data.allocations[i].followup_count;
    allocation+=path(`M ${x-65} 283 H ${x+65}`,'#EEE9F2',7);
    if(n)allocation+=path(`M ${x-65} 283 H ${x-65+130*n/17}`,colors.alloc,7,`class="framework-allocation" data-bar-start="${x-65}" data-bar-y="283" data-bar-width="${130*n/17}"`);
    allocation+=text(x,312,`${n} extra`,18,colors.alloc,'middle',n?600:400);
  });
  drawing += group(3,allocation);

  let followups='';
  xs.forEach((x,i)=>{
    const n=data.allocations[i].followup_count;
    if(n){
      followups+=flow(4,`M ${x} 319 V 331`,colors.code);
      for(let j=0;j<n;j++){
        const row=Math.floor(j/9),cols=Math.min(9,n-row*9);
        followups+=circle(x+(j%9-(cols-1)/2)*11,337+row*12,3.5,colors.code);
      }
      followups+=score(x,375,data.allocations[i].final_best_raw_score,i===2?colors.red:colors.eval);
    }
    followups+=flow(4,`M ${x} ${n?382:321} V 394`,colors.eval);
  });
  followups+=path('M 375 394 H 1035',colors.eval,1.4)+flow(4,'M 584 394 V 407',colors.eval);
  followups+=box(415,407,335,43,colors.red,'#FCF0F1')+text(582.5,435,'SPO² scoring + backward',19,colors.red,'middle',600);
  drawing += group(4,followups);

  let update=flow(5,'M 750 429 H 820',colors.red)+box(820,407,230,43,colors.red,'#FCF0F1')+text(935,435,'One optimizer update',19,colors.red,'middle',600);
  update+=flow(5,'M 935 451 V 465 H 250 V 123 H 263',colors.red);
  update+=flow(5,'M 815 382 V 387 H 168 V 187',colors.red);
  update+=circle(168,180,7,colors.red)+path('M 181 118 H 217 V 180 H 177',colors.red,2);
  update+=text(168,207,'Retained child',17,colors.red,'middle',600)+score(168,229,data.allocationFinalBest,colors.red);
  update+=path('M 257 118 L 263 123 L 257 128',colors.red,1.5)+path('M 812 424 L 820 429 L 812 434',colors.red,1.5);
  update+=text(275,455,'Next step',15,colors.red);
  drawing += group(5,update);

  host.innerHTML = `<svg class="data-figure framework-figure" viewBox="0 0 1156 480" role="img" aria-label="Six stages of hierarchical scientific discovery. A scored archive supplies parents. A frozen strategist proposes plans, and a shared coder generates equal pilots. Pilot scores allocate follow-ups. Evaluated programs train the coder with SPO squared. Selected children return to the archive and one optimizer update precedes the next step."><title>Hierarchical scientific discovery</title>${drawing}</svg>`;
  const phaseTitle = host.closest('.slide')?.querySelector('.framework-stage-title') || document.getElementById('framework-stage-title');
  const phaseCount = host.closest('.slide')?.querySelector('.framework-stage-count') || document.getElementById('framework-stage-count');
  const phaseCaption = host.closest('.slide')?.querySelector('.framework-caption') || document.getElementById('framework-caption');
  const fragments=[...host.querySelectorAll('[data-framework-phase]')];
  const traces=[...host.querySelectorAll('.framework-trace')];
  const bars=[...host.querySelectorAll('.framework-allocation')];
  const ease=t=>1-Math.pow(1-Math.max(0,Math.min(1,t)),3);
  let stage=5;
  function setStage(next,progress=1,moving=false) {
    stage=Math.max(0,Math.min(5,next));
    if(phaseTitle)phaseTitle.textContent=phases[stage][0];
    if(phaseCount)phaseCount.textContent=`${String(stage+1).padStart(2,'0')} / 06`;
    if(phaseCaption)phaseCaption.textContent=phases[stage][1];
    fragments.forEach(g=>{
      const phase=Number(g.dataset.frameworkPhase),t=phase===stage?ease(progress*3):1;
      g.style.opacity=phase>stage?'0':String(t);
      g.style.visibility=phase>stage?'hidden':'visible';
      g.setAttribute('transform',`translate(0 ${phase===stage?(1-t)*7:0})`);
    });
    host.querySelectorAll('[data-through-phase]').forEach(g=>{g.style.visibility=stage>Number(g.dataset.throughPhase)?'hidden':'visible';});
    host.querySelectorAll('[data-framework-stagger]').forEach(g=>{
      const t=stage===1?ease(progress*2.6-Number(g.dataset.frameworkStagger)*.22):1;
      g.style.opacity=String(t);
      g.setAttribute('transform',`translate(0 ${(1-t)*5})`);
    });
    bars.forEach(bar=>{
      const t=stage===3?ease(progress*1.8):1,x=Number(bar.dataset.barStart),y=Number(bar.dataset.barY);
      bar.setAttribute('d',`M ${x} ${y} H ${x+Number(bar.dataset.barWidth)*t}`);
    });
    traces.forEach(trace=>{
      const active=moving&&Number(trace.dataset.flowPhase)===stage&&progress>.12&&progress<.98;
      trace.style.opacity=active?'.9':'0';
      trace.setAttribute('stroke-dasharray','.075 .925');
      trace.setAttribute('stroke-dashoffset',String(-((progress*1.7)%1)));
    });
    const ring=host.querySelector('.framework-parent-ring');
    ring.setAttribute('r',moving&&stage===0?String(13+2*Math.sin(progress*Math.PI*2)):13);
    return stage;
  }
  function seek(seconds) {
    const t=Math.max(0,Math.min(total,seconds));
    const i=starts.findLastIndex(start=>t>=start);
    return setStage(i,Math.min(1,(t-starts[i])/durations[i]),true);
  }
  setStage(5);
  return {setStage,seek,total,phases,starts,get stage(){return stage;}};
}

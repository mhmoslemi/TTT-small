'use strict';

function createFramework(host,data){
  const C={ink:'#202328',grey:'#66717D',pale:'#D7DEE6',blue:'#466B8A',teal:'#25817C',gold:'#A77526',purple:'#806096',red:'#C7000B'};
  const phases=[
    ['A search for a better mathematical solution',''],
    ['Each strategy uses the parent and every earlier strategy',''],
    ['The coder turns each strategy into testable candidates',''],
    ['The bandit allocates more attempts using pilot results',''],
    ['One SPO² update from all evaluated rollouts',''],
    ['The next step uses a saved child and the updated coder',''],
  ];
  const durations=[7,11,9,8,30,9],starts=durations.map((_,i)=>durations.slice(0,i).reduce((a,b)=>a+b,0));
  const total=durations.reduce((a,b)=>a+b,0),trainingTimes=[48.5];
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,s,size=20,color=C.ink,anchor='middle',extra='')=>`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" style="fill:${color}" ${extra}>${esc(s)}</text>`;
  const dot=(x,y,r,fill,stroke=fill,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5" ${extra}/>`;
  const line=(d,color=C.pale,width=1.5,extra='')=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const group=(id,body)=>`<g data-scene="${id}">${body}</g>`;
  const curve=(a,b,bend=0)=>{
    const points=Array.from({length:17},(_,i)=>{const t=i/16,u=1-t;return [u*a[0]+t*b[0],u*a[1]+t*b[1]-4*t*u*bend];});
    return points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ');
  };
  // Vertical tangents meet node centers; drawing nodes last hides the joins.
  const branch=(a,b)=>Array.from({length:33},(_,i)=>{const t=i/32,s=t*t*(3-2*t);return (i?'L':'M')+[a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*t].join(' ');}).join(' ');
  const xs=[150,435,720,1005],selected=[555,265],parent=[578,38];
  let forestEdges='',forestNodes='',order=0;
  [140,430,730,1020].forEach((x,r)=>{
    const tree=`data-forest-tree="${r}"`;
    forestNodes+=dot(x,60,6,r===1?C.red:'#B3BDC9',undefined,tree);
    [-80,80].forEach((dx,k)=>{
      const bx=x+dx,ancestor=r===1&&k===1;
      forestEdges+=line(branch([x,60],[bx,155]),ancestor?C.red:C.pale,ancestor?2:1.5,tree);
      forestNodes+=dot(bx,155,5.5,ancestor?C.red:'#CAD3DD',undefined,tree);
      [-45,0,45].forEach((dd,j)=>{
        const lx=bx+dd,chosen=ancestor&&j===2;
        forestEdges+=line(branch([bx,155],[lx,265]),chosen?C.red:C.pale,chosen?2:1.5,tree);
        if(!chosen)forestNodes+=dot(lx,265,4.7,'#DDE3E9','#C8D1DB',`data-forest-order="${order++}" ${tree}`);
      });
    });
  });
  let drawing=group('forest',forestEdges+forestNodes);
  drawing+=group('archiveHelp',text(0,20,'In this Erdős example, lower scores are better.',20,C.grey,'start')+text(578,375,'Each dot is a saved candidate. Each link records where it came from.',23,C.ink)+text(578,415,'A candidate is a mathematical construction produced by a program.',21,C.grey));

  let expansion='';
  xs.forEach((x,i)=>{
    expansion+=`<g data-plan="${i}">${line(branch(parent,[x,100]),C.blue,1.6,`data-grow-plan="${i}" pathLength="1"`)}<g data-strategy-label>${dot(x,100,8,C.blue)}${text(x,133,`Strategy ${i+1}`,21,C.blue,'middle','font-weight="600"')}</g></g>`;
  });
  drawing+=group('plans',expansion);
  const strategyYs=[105,193,281,369];
  let planning=text(1010,55,'Generated strategies',20,C.blue);
  planning+=`<g data-strategist-model><rect x="520" y="181" width="190" height="88" rx="5" fill="#F3F6F8" stroke="${C.blue}" stroke-width="1.4"/>${text(615,233,'Strategist',26,C.blue,'middle','font-weight="600"')}${text(615,299,'Frozen weights',18,C.grey)}</g>`;
  for(let i=0;i<4;i++){
    const y=strategyYs[i];
    let inputs=text(38,55,`Prompt for Strategy ${i+1}`,22,C.ink,'start','data-prompt-title font-weight="600"')+dot(38,122,8,C.red,C.red,'data-planning-parent')+text(60,129,'Task + parent',22,C.red,'start');
    inputs+=line('M 38 150 H 370',C.pale,1)+text(38,180,i?'All earlier strategies':'No earlier strategies yet',18,C.grey,'start','data-history-label');
    for(let j=0;j<i;j++)inputs+=`<g data-prior-strategy="${j}">${dot(54,205+j*58,6,C.blue)}${text(72,212+j*58,`Strategy ${j+1}`,22,C.blue,'start','font-weight="600"')}</g>`;
    inputs+=line('M 389 97 H 412 V 350 H 389',C.pale,1.4);
    inputs+=line('M 412 225 H 504',C.blue,1.6,'data-call-in pathLength="1"')+line('M 497 220 L 504 225 L 497 230',C.blue,1.6,'data-call-in-head');
    planning+=`<g data-strategy-call="${i}"><g data-call-inputs>${inputs}</g>`;
    planning+=line(curve([730,225],[925,y]),C.blue,1.6,'data-call-out pathLength="1"');
    planning+=`<g data-call-result>${dot(925,y,6,C.blue)}${text(947,y+7,`Strategy ${i+1}`,22,C.blue,'start','font-weight="600"')}</g></g>`;
  }
  for(let i=0;i<3;i++)planning+=`<g data-history-transfer="${i}">${dot(925,strategyYs[i],6,C.blue)}${text(947,strategyYs[i]+7,`Strategy ${i+1}`,22,C.blue,'start','font-weight="600"')}</g>`;
  drawing+=group('planning',planning);

  let pilots='';
  const pilotPositions=[];
  xs.forEach((x,i)=>{
    pilots+=`<g data-coder="${i}">${line(`M ${x} 140 V 157 M ${x-4} 153 L ${x} 157 L ${x+4} 153`,C.teal,1.5,'data-coder-input')}`;
    pilots+=`<rect x="${x-70}" y="161" width="140" height="36" rx="4" fill="#F1F7F6" stroke="${C.teal}" stroke-width="1.3"/>${text(x,185,'Shared coder',18,C.teal,'middle','font-weight="600"')}</g>`;
    pilots+=line(`M ${x} 201 V 224 M ${x-4} 220 L ${x} 224 L ${x+4} 220`,C.teal,1.5,`data-coder-output="${i}"`);
    for(let j=0;j<11;j++){
      const px=x+(j-5)*12,py=241;pilotPositions.push([px,py]);
      pilots+=dot(px,py,4.6,C.teal,C.teal,`data-pilot="${i*11+j}" data-pilot-valid="${j<data.allocations[i].pilot_valid_count}"`);
    }
    pilots+=text(x,269,'11 trial programs',17,C.teal,'middle',`data-pilot-count="${i}"`);
    pilots+=`<g data-pilot-score="${i}">${line(`M ${x-67} 278 H ${x+67}`,C.gold,1.3)}${text(x,309,data.allocations[i].pilot_best_raw_score.toFixed(6),20,C.gold)}</g>`;
  });
  pilots+=text(0,309,'Best score',16,C.gold,'start','data-pilot-score-label');
  drawing+=group('pilots',pilots);
  drawing+=group('evaluationHelp',text(578,363,'Each dot is one program',23,C.ink,'middle','font-weight="600"')+dot(365,397,5,C.teal)+text(383,403,'Valid output',19,C.teal,'start')+dot(650,397,5,'#E0E5EB','#CDD5DF')+text(668,403,'Invalid output',19,C.grey,'start')+text(578,450,'The evaluator runs the code, checks its output, and assigns a score.',21,C.ink));

  let followups=text(578,411,'20 extra attempts to allocate',21,C.purple,'middle','data-pool-label="true"');
  const extraPositions=[],extraStarts=[];
  for(let j=0;j<20;j++){
    const lane=j<17?2:3,index=j<17?j:j-17,n=lane===2?17:3,row=Math.floor(index/9),cols=Math.min(9,n-row*9);
    const target=[xs[lane]+(index%9-(cols-1)/2)*13,353+row*14];extraPositions.push(target);
    const start=[target[0]-xs[lane]+(lane===2?540:655),target[1]];extraStarts.push(start);
    followups+=dot(...start,4.7,C.purple,C.purple,`data-extra="${j}"`);
  }
  xs.forEach((x,i)=>{followups+=text(x,399,`${data.allocations[i].followup_count} extra attempts`,19,i>1?C.purple:C.grey,'middle',`data-extra-count="${i}" font-weight="${i>1?600:400}"`);});
  followups+=`<g data-child-score="true">${text(590,450,'Best new score',20,C.red,'end')}${dot(720,423,5,C.red,C.red,'data-winning-child')}${text(720,455,data.allocationFinalBest.toFixed(6),26,C.red,'middle','data-winning-score font-weight="600"')}${text(1005,454,data.allocations[3].final_best_raw_score.toFixed(6),18,C.gold)}</g>`;
  drawing+=group('followups',followups);

  // Illustrative rewards, not measurements. The reference follows the count-weighted tracker.
  const rewards=[.1,.2,.3,.4,.5,.6,.7,.8],pastMeans=[1.3,5.8,6.2,3],history=[pastMeans[0]];
  const pastRewards=pastMeans.flatMap(mean=>[.85,.9,.95,.98,1.02,1.05,1.1,1.15].map(factor=>Math.log(mean*factor)/2));
  let effectiveCount=8/(1-.875);
  for(const mean of pastMeans.slice(1)){
    history.push((.9*effectiveCount*history.at(-1)+8*mean)/(.9*effectiveCount+8));effectiveCount=.9*effectiveCount+8;
  }
  const historicalReference=history.at(-1),advantages=rewards.map(r=>(Math.exp(2*r)/historicalReference-1)/2);
  const groupMeans=[.25,.65],groupStd=Math.sqrt(.0125),grpoAdvantages=rewards.map((r,i)=>(r-groupMeans[Math.floor(i/4)])/groupStd);
  const rewardX=(r,method=0)=>110+600*method+400*r,referenceX=rewardX(Math.log(historicalReference)/2);
  let learning=text(578,25,'Illustrative rewards · higher is better',17,C.grey,'middle','data-training-example');
  for(let method=0;method<2;method++){
    const offset=method*600;
    let chart=text(280+offset,55,method?'Standard GRPO':'SPO²',26,method?C.blue:C.red,'middle',`font-weight="600" ${method?'':'data-training-heading'}`);
    chart+=text(280+offset,84,method?'A separate reference for each group':'One reference from earlier rollouts',19,C.grey);
    for(let row=0;row<2;row++){
      const y=220+row*100;
      chart+=text(16+offset,y+6,`Group ${row?'B':'A'}`,18,C.ink,'start');
      chart+=line(`M ${110+offset} ${y} H ${510+offset}`,C.pale,1);
      if(method){
        const x=rewardX(groupMeans[row],1);
        chart+=`<g data-group-reference="${row}">${line(`M ${x} ${y-40} V ${y+35}`,C.blue,1.5,'stroke-dasharray="4 4"')}${text(x,y-51,'Group mean',18,C.blue)}</g>`;
      }
    }
    chart+=line(`M ${110+offset} 370 H ${510+offset} M ${504+offset} 366 L ${510+offset} 370 L ${504+offset} 374`,C.grey,1.1);
    for(const r of [0,.5,1])chart+=text(rewardX(r,method),395,String(r),16,C.grey);
    chart+=text(310+offset,420,'Reward',18,C.grey);
    rewards.forEach((r,i)=>{
      const x=rewardX(r,method),y=220+100*Math.floor(i/4),a=(method?grpoAdvantages:advantages)[i];
      chart+=`<g data-${method?'grpo':'spo'}-rollout="${i}" data-reward="${r}" data-advantage="${a}">${line('',C.grey,2,'data-learning-arrow')}${dot(x,y,6.5,C.teal,C.teal,'data-learning-dot')}</g>`;
    });
    learning+=`<g data-training-${method?'grpo':'spo'}>${chart}</g>`;
  }
  let historyView=text(280,119,'Earlier rollouts',18,C.blue);
  pastRewards.forEach((reward,i)=>{
    historyView+=dot(rewardX(reward),133+i%4*8,3.8,C.blue,C.blue,`data-history-rollout="${i}" data-reward="${reward}"`);
  });
  learning+=`<g data-history-inputs>${historyView}</g>`;
  learning+=`<g data-history-reference>${line(`M ${referenceX} 183 V 365`,C.blue,1.7,'data-baseline-line stroke-dasharray="5 5"')}${dot(referenceX,183,5.5,C.blue)}${text(referenceX+16,189,'Shared reference',18,C.blue,'start','data-baseline-label')}${text(referenceX+16,209,'Moving average of amplified rewards',16,C.grey,'start','data-reference-status')}</g>`;
  learning+=text(650,106,'Current rollouts',17,C.grey,'middle','data-rollout-label');
  learning+=text(578,463,'Direction only: ↑ reinforce   ↓ discourage',20,C.ink,'middle','data-training-legend');
  learning+=`<g data-weight-explanation>${text(900,200,'Exponential weighting',25,C.red,'middle','font-weight="600"')}${text(900,240,'Exceptional rewards receive',23,C.ink)}${text(900,272,'progressively larger weights',23,C.ink)}</g>`;
  const coderBody=`${text(930,107,'Coding model',24,C.ink,'middle','font-weight="600"')}<rect x="800" y="137" width="260" height="113" rx="4" fill="#F5F7F9" stroke="${C.pale}" stroke-width="1.5"/>${text(930,182,'Base weights',23,C.ink)}${text(930,216,'Frozen',19,C.grey)}<rect x="800" y="258" width="260" height="58" rx="4" fill="#FFF6F6" stroke="${C.red}" stroke-width="1.5" data-adapter-box/>${text(820,293,'LoRA',22,C.red,'start','font-weight="600"')}${text(1040,293,'Trainable',19,C.red,'end','data-adapter-status')}`;
  learning+=`<g data-learning-model><g data-coder-body>${coderBody}</g>${text(930,355,'One optimizer update',23,C.ink,'middle','font-weight="600"')}${text(930,390,'The strategist stays frozen',20,C.blue)}</g>`;
  drawing+=group('learning',learning);
  const rolloutPositions=[...pilotPositions,...extraPositions];
  const gatheredPosition=i=>[650+(i%16-7.5)*9,126+Math.floor(i/16)*11];
  let outcomes='';
  rolloutPositions.forEach(([x,y],i)=>{
    const invalid=i<44&&!(i%11<data.allocations[Math.floor(i/11)].pilot_valid_count);
    outcomes+=dot(x,y,3.4,invalid?'#E0E5EB':i<44?C.teal:C.purple,invalid?'#CDD5DF':undefined,`data-outcome="${i}"`);
  });
  drawing+=group('outcomes',outcomes);

  let next=line(branch([407,197],[480,280]),C.red,2,'data-next-retained-edge')+dot(407,197,5.5,C.red);
  next+=text(450,278,'Saved child',20,C.red,'end','data-next-parent-label font-weight="600"')+text(450,307,data.allocationFinalBest.toFixed(6),22,C.red,'end','data-next-score');
  next+=dot(480,280,6,C.red,C.red,'data-next-child');
  next+=`<g data-next-model>${coderBody.replace('Trainable','Updated').replace('#FFF6F6','#FBE8EA').replace('data-adapter-box','data-next-adapter-box').replace('data-adapter-status','data-next-adapter-status')}</g>`;
  next+=line('M 493 280 H 746 Q 760 280 760 266 V 215 Q 760 201 774 201 H 784',C.blue,1.6,'data-next-input pathLength="1"')+line('M 777 196 L 784 201 L 777 206',C.blue,1.6,'data-next-input-head')+text(635,253,'Parent + new strategies',18,C.blue,'middle','data-next-input-label');
  [410,480,550].forEach((x,i)=>{next+=line(branch([480,280],[x,415]),C.teal,1.5,`data-new-branch="${i}" pathLength="1"`)+dot(x,415,6,'#FFFFFF',C.teal,`data-new-node="${i}"`);});
  next+=text(480,455,'New programs',21,C.teal,'middle','data-new-programs-label')+text(0,24,'Saved candidates',23,C.ink,'start','font-weight="600"');
  drawing+=group('next',next);
  drawing+=group('parent',dot(0,0,11,C.red)+text(0,33,'Starting candidate (parent)',20,C.ink,'middle','data-parent-name="true" font-weight="600"')+text(0,59,`Score: ${data.lineage[0].score.toFixed(6)}`,18,C.grey,'middle','data-parent-score="true"'));

  host.innerHTML=`<svg class="data-figure framework-figure" viewBox="0 0 1156 480" role="img" aria-label="Animated search archive and one parent expansion. Each strategist call receives the task, parent and every earlier strategy for that parent. Four strategies receive equal pilots, then 20 follow-ups are allocated as zero, zero, 17 and 3. All evaluated rollouts contribute to one SPO squared update. A comparison uses the same eight illustrative rewards in two input groups: standard GRPO centers within each group, while SPO squared uses one historical reference. Exponential weighting increases the influence of exceptional rewards. The next step uses the updated coder and saved candidates."><title>Hierarchical scientific discovery</title><g class="framework-layout">${drawing}</g><g class="framework-transition" aria-hidden="true"></g></svg>`;
  const layout=host.querySelector('.framework-layout'),transition=host.querySelector('.framework-transition');
  const groups=Object.fromEntries([...host.querySelectorAll('[data-scene]')].map(g=>[g.dataset.scene,g]));
  const title=host.closest('.slide')?.querySelector('.framework-stage-title')||document.getElementById('framework-stage-title');
  const count=host.closest('.slide')?.querySelector('.framework-stage-count')||document.getElementById('framework-stage-count');
  const caption=host.closest('.slide')?.querySelector('.framework-caption')||document.getElementById('framework-caption');
  const sceneElements=[host,title,count,caption];
  const clamp=t=>Math.max(0,Math.min(1,t)),ease=t=>{t=clamp(t);return t*t*(3-2*t);},mix=(a,b,t)=>a+(b-a)*t;
  const opacity=(e,v)=>{e.style.opacity=String(clamp(v));e.style.visibility=v>0?'visible':'hidden';};
  const move=(e,x,y)=>e.setAttribute('transform',`translate(${x} ${y})`);
  let stage=5;
  function setStage(s,progress=1,moving=false){
    stage=Math.max(0,Math.min(5,s));const p=clamp(progress);
    opacity(layout,1);opacity(transition,0);
    sceneElements.forEach(e=>e.style.opacity='1');
    title.style.transform='none';caption.style.transform='none';
    title.textContent=phases[stage][0];count.textContent='';caption.textContent=phases[stage][1];
    Object.values(groups).forEach(g=>opacity(g,0));
    groups.forest.setAttribute('transform',stage===5?'translate(35 20) scale(.67)':'translate(0 0)');
    groups.forest.querySelectorAll('[data-forest-tree]').forEach(e=>opacity(e,stage===5&&e.dataset.forestTree==='3'?0:1));
    const root=stage===0?selected:stage===5?[407,197]:parent;
    move(groups.parent,...root);opacity(groups.parent,stage===1||stage===4?0:1);
    const name=host.querySelector('[data-parent-name]'),value=host.querySelector('[data-parent-score]');
    name.textContent=stage===0?'Starting candidate (parent)':'Parent';name.setAttribute('x',stage===0?0:-26);name.setAttribute('y',stage===0?33:6);name.setAttribute('text-anchor',stage===0?'middle':'end');
    value.setAttribute('x',stage===0?0:26);value.setAttribute('y',stage===0?59:6);value.setAttribute('text-anchor',stage===0?'middle':'start');
    opacity(name,stage===5?0:1);opacity(value,stage===5?0:1);
    if(stage===0){opacity(groups.forest,ease(p*3));opacity(groups.archiveHelp,ease((p-.15)*3));}
    if(stage===1)opacity(groups.planning,1);
    const callIndex=Math.min(3,Math.floor(p*4)),callProgress=p*4-callIndex;
    host.querySelectorAll('[data-strategy-call]').forEach(g=>{
      const i=Number(g.dataset.strategyCall),a=p*4-i,active=i===callIndex,arrived=ease((a-.2)/.22);
      opacity(g,1);opacity(g.querySelector('[data-call-inputs]'),active?1:0);
      const preparing=i<3&&a>=.54;
      g.querySelector('[data-prompt-title]').textContent=preparing?`Next prompt: Strategy ${i+2}`:`Prompt for Strategy ${i+1}`;
      g.querySelector('[data-history-label]').textContent=i||preparing?'All earlier strategies':'No earlier strategies yet';
      const result=g.querySelector('[data-call-result]'),node=result.querySelector('circle'),label=result.querySelector('text');
      opacity(result,ease((a-.2)/.07));opacity(label,ease((a-.42)/.08));
      node.setAttribute('cx',mix(730,925,arrived));node.setAttribute('cy',mix(225,strategyYs[i],arrived));
      const input=g.querySelector('[data-call-in]'),output=g.querySelector('[data-call-out]');
      opacity(input,i<3?1-ease((a-.5)/.04):1);
      opacity(g.querySelector('[data-call-in-head]'),ease((a-.16)/.04)*(i<3?1-ease((a-.5)/.04):1));
      input.setAttribute('stroke-dasharray','1');input.setAttribute('stroke-dashoffset',String(1-ease((a-.03)/.13)));
      output.setAttribute('stroke-dasharray','1');output.setAttribute('stroke-dashoffset',String(1-arrived));opacity(output,active&&a>.2?1-ease((a-.48)/.14):0);
    });
    const model=host.querySelector('[data-strategist-model] rect'),pulse=Math.sin(Math.PI*clamp((callProgress-.12)/.38));
    model.setAttribute('fill',`rgb(${mix(243,224,pulse)},${mix(246,235,pulse)},${mix(248,242,pulse)})`);
    host.querySelectorAll('[data-history-transfer]').forEach(g=>{
      const i=Number(g.dataset.historyTransfer),a=p*4-i,t=ease((a-.54)/.46),u=1-t,y=strategyYs[i]+7;
      opacity(g,a>.54&&a<1?1:0);
      const x=u*u*u*947+3*u*u*t*1050+3*u*t*t*72+t*t*t*72;
      const cy=u*u*u*y+3*u*u*t*430+3*u*t*t*430+t*t*t*(212+i*58);
      const label=g.querySelector('text'),node=g.querySelector('circle');
      label.setAttribute('x',x);label.setAttribute('y',cy);node.setAttribute('cx',x-mix(22,18,t));node.setAttribute('cy',cy-7);
      opacity(label,ease((Math.hypot(x-947,cy-y)-32)/30));
    });
    if(stage===2||stage===3){opacity(groups.plans,1);opacity(groups.pilots,1);}
    if(stage===2)opacity(groups.evaluationHelp,ease((p-.45)*3));
    host.querySelectorAll('[data-plan]').forEach(g=>{const k=Number(g.dataset.plan);opacity(g,stage===1?ease((p-.2)*3-k*.24):1);});
    host.querySelectorAll('[data-grow-plan]').forEach(e=>{e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',stage===1?String(1-ease((p-.2)*3-Number(e.dataset.growPlan)*.24)):'0');});
    host.querySelectorAll('[data-coder]').forEach(e=>opacity(e,stage===2?ease(p/.18):1));
    host.querySelectorAll('[data-coder-output]').forEach(e=>opacity(e,stage===2?ease((p-.18)/.12):1));
    host.querySelectorAll('[data-pilot]').forEach(e=>{
      const i=Number(e.dataset.pilot),j=i%11,[x,y]=pilotPositions[i],a=stage===2?ease((p-.3)/.18):1;
      e.setAttribute('cx',x);e.setAttribute('cy',y);opacity(e,a);
      const evaluated=stage>2||p>.58+j*.025,valid=e.dataset.pilotValid==='true';
      e.setAttribute('fill',evaluated&&!valid?'#E0E5EB':C.teal);e.setAttribute('stroke',evaluated&&!valid?'#CDD5DF':C.teal);
    });
    host.querySelectorAll('[data-pilot-score]').forEach(e=>opacity(e,stage===2?ease((p-.65)*4):1));
    opacity(host.querySelector('[data-pilot-score-label]'),stage===2?ease((p-.65)*4):1);
    host.querySelectorAll('[data-pilot-count]').forEach(e=>opacity(e,stage===2?ease((p-.48)/.12):1));
    if(stage===3)opacity(groups.followups,1);
    host.querySelectorAll('[data-extra]').forEach(e=>{
      const i=Number(e.dataset.extra),a=stage===3?ease((p-.18)/.42):1,[x,y]=extraPositions[i],[startX,startY]=extraStarts[i];
      e.setAttribute('cx',mix(startX,x,a));e.setAttribute('cy',mix(startY,y,a));opacity(e,stage===3?ease(p*7):1);
    });
    opacity(host.querySelector('[data-pool-label]'),stage===3?1-ease((p-.15)*4):0);
    host.querySelectorAll('[data-extra-count]').forEach(e=>opacity(e,stage===3?ease((p-.55)*4):1));
    opacity(host.querySelector('[data-child-score]'),stage===3?ease((p-.83)*6):1);
    if(stage===4)showUpdate(1.6+p*27.4);
    if(stage===5){
      opacity(groups.forest,1);opacity(groups.next,1);
      const selection=ease((p-.08)/.22);
      move(groups.parent,407+73*ease(selection),197+83*selection);
      const label=host.querySelector('[data-next-parent-label]');label.textContent=selection<.5?'Saved child':'Next parent';opacity(label,selection<.5?1-ease(selection*2):ease((selection-.5)*2));
    }
    const input=stage===5?ease((p-.34)/.15):0,inputLine=host.querySelector('[data-next-input]');
    inputLine.setAttribute('stroke-dasharray','1');inputLine.setAttribute('stroke-dashoffset',String(1-input));opacity(inputLine,input>0?1:0);
    opacity(host.querySelector('[data-next-input-label]'),ease(input*3));
    opacity(host.querySelector('[data-next-input-head]'),stage===5?ease((p-.49)/.01):0);
    host.querySelectorAll('[data-new-branch]').forEach(e=>{const a=stage===5?ease((p-.65-Number(e.dataset.newBranch)*.08)/.11):0;e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',String(1-a));opacity(e,a>0?1:0);});
    host.querySelectorAll('[data-new-node]').forEach((e,i)=>{
      const a=stage===5?ease((p-.58-i*.08)/.18):0;
      e.setAttribute('cx',mix(930,410+i*70,a));e.setAttribute('cy',mix(324,415,a)+Math.sin(a*Math.PI)*32);opacity(e,ease(a*8));
    });
    opacity(host.querySelector('[data-new-programs-label]'),stage===5?ease((p-.76)/.12):0);
    return stage;
  }
  const svgGroup=()=>document.createElementNS('http://www.w3.org/2000/svg','g');
  function showUpdate(local){
    opacity(layout,1);opacity(transition,0);opacity(groups.learning,1);opacity(groups.outcomes,1);opacity(groups.parent,0);
    const weight=ease((local-16.7)/2.8);
    host.querySelectorAll('[data-outcome]').forEach((e,i)=>{
      const [x,y]=gatheredPosition(i),a=ease((local-23-(15-i%16)/15*1.25)/2.3),u=1-a,targetY=270+Math.floor(i/16)*11;
      e.setAttribute('cx',u*u*u*x+3*u*u*a*752+3*u*a*a*745+a*a*a*800);
      e.setAttribute('cy',u*u*u*y+3*u*u*a*y+3*u*a*a*targetY+a*a*a*targetY);opacity(e,1-ease((a-.95)/.05));
    });
    opacity(host.querySelector('[data-rollout-label]'),ease((local-3.1)/.6)*(1-ease((local-24.5)/.6)));
    opacity(host.querySelector('[data-training-example]'),ease((local-3.1)/.6));
    opacity(host.querySelector('[data-training-spo]'),ease((local-3.1)/.6));
    const heading=host.querySelector('[data-training-heading]');heading.textContent='SPO²';heading.setAttribute('font-size',26);
    opacity(host.querySelector('[data-training-grpo]'),ease((local-3.1)/.6)*(1-ease((local-15.7)/.8)));
    // Past rewards arrive before each historical update; current advantages use the settled value.
    let reference=history[0];
    for(let i=1;i<history.length;i++)reference+=(history[i]-history[i-1])*ease((local-(5.15+(i-1)*1.4))/.6);
    const x=rewardX(Math.log(reference)/2);
    const ref=host.querySelector('[data-history-reference]');move(ref,x-referenceX,0);ref.dataset.referenceValue=String(reference);
    opacity(ref,ease((local-4.1)/.5));
    const status=host.querySelector('[data-reference-status]');status.textContent=local<8.7?'Moving average of amplified rewards':'Fixed for this step';
    opacity(status,local<8.7?1-ease((local-8.55)/.15):ease((local-8.7)/.15));
    opacity(host.querySelector('[data-history-inputs]'),ease((local-3.4)/.5));
    host.querySelectorAll('[data-history-rollout]').forEach((e,i)=>{
      const step=Math.floor(i/8),arrival=step===0?3.4:4.2+(step-1)*1.4,entered=ease((local-arrival-i%8*.06)/.3);
      opacity(e,entered*(1-.45*ease((local-8.55)/.5)));
    });
    host.querySelectorAll('[data-group-reference]').forEach(e=>opacity(e,ease((local-9.6)/.5)));
    for(let method=0;method<2;method++)host.querySelectorAll(`[data-${method?'grpo':'spo'}-rollout]`).forEach((g,i)=>{
      const x=rewardX(rewards[i],method),y=220+100*Math.floor(i/4),a=(method?grpoAdvantages:advantages)[i],direction=Math.sign(a);
      const arrow=g.querySelector('[data-learning-arrow]'),point=g.querySelector('[data-learning-dot]');
      const length=method?25:mix(25,70*Math.abs(a)/advantages.at(-1),weight),tip=y-direction*(7+length),color=direction<0?C.grey:!method&&i===7&&weight>.5?C.red:C.teal;
      const head=Math.min(5,length*.65),halfWidth=Math.min(4,head*.8);
      opacity(g,ease((local-9-i*.05)/.65));opacity(arrow,ease((local-10.1)/.65));
      arrow.setAttribute('d',`M ${x} ${y-direction*7} V ${tip} M ${x-halfWidth} ${tip+direction*head} L ${x} ${tip} L ${x+halfWidth} ${tip+direction*head}`);arrow.setAttribute('stroke',color);arrow.setAttribute('stroke-width',!method?2+weight:2);
      point.setAttribute('fill',color);point.setAttribute('stroke',color);
    });
    opacity(host.querySelector('[data-weight-explanation]'),ease((local-16.6)/.7)*(1-ease((local-20.6)/.7)));
    opacity(host.querySelector('[data-learning-model]'),ease((local-21.4)/.8));
    const legend=host.querySelector('[data-training-legend]');legend.textContent=local<16?'Direction only: ↑ reinforce   ↓ discourage':'SPO² arrow length shows the learning weight; every rollout contributes.';
    opacity(legend,ease((local-10.1)/.65));
    const sections=[1.6,9,16,21.4],titles=['Earlier rollout rewards set the shared reference','Training signals for the same eight rewards','SPO² gives exceptional rewards more influence',phases[4][0]];
    const section=Math.max(0,sections.findLastIndex(t=>local>=t)),end=sections[section+1]??30;
    title.textContent=titles[section];title.style.opacity=String((section===0?1:ease((local-sections[section])/.35))*(section===3?1:1-ease((local-end+.35)/.35)));
    const updated=ease((local-27)/.6),adapter=host.querySelector('[data-adapter-box]');
    host.querySelector('[data-adapter-status]').textContent=local<27?'Trainable':'Updated';
    adapter.setAttribute('fill',`rgb(${mix(255,251,updated)},${mix(246,232,updated)},${mix(246,234,updated)})`);
    adapter.setAttribute('stroke-width',String(1.5+1.5*Math.sin(Math.PI*updated)));
    caption.textContent='';stage=4;
  }
  const snapshot=(s,p)=>{setStage(s,p);const copy=layout.cloneNode(true);copy.querySelectorAll('[data-scene]').forEach(g=>{if(g.style.opacity==='0')g.remove();});return copy;};
  const hide=(copy,selector)=>copy.querySelectorAll(selector).forEach(e=>opacity(e,0));
  const bridges=[];
  for(let i=0;i<5;i++){
    const out=snapshot(i,1),incoming=snapshot(i+1,0),keep=svgGroup(),anchors=svgGroup(),container=svgGroup();
    if(i===0){
      const parentCopy=out.querySelector('[data-scene=parent]').cloneNode(true);
      anchors.append(parentCopy);hide(out,'[data-scene=parent]');hide(incoming,'[data-planning-parent]');
    }else if(i===1){
      for(let j=0;j<4;j++)anchors.innerHTML+=`<g>${dot(925,strategyYs[j],6,C.blue)}${text(947,strategyYs[j]+7,`Strategy ${j+1}`,22,C.blue,'start','font-weight="600"')}</g>`;
      anchors.innerHTML+=dot(38,122,8,C.red);
      hide(out,'[data-call-result],[data-planning-parent]');hide(incoming,'[data-strategy-label],[data-scene=parent] circle');
    }else if(i===2){
      for(const id of ['plans','pilots','parent']){
        keep.append(out.querySelector(`[data-scene=${id}]`).cloneNode(true));
        hide(out,`[data-scene=${id}]`);hide(incoming,`[data-scene=${id}]`);
      }
    }else if(i===3){
      out.querySelectorAll('[data-pilot],[data-extra]').forEach(e=>anchors.append(e.cloneNode(true)));
      hide(out,'[data-pilot],[data-extra]');hide(incoming,'[data-scene=outcomes],[data-scene=learning],[data-scene=parent]');
    }else{
      anchors.append(out.querySelector('[data-coder-body]').cloneNode(true));
      hide(out,'[data-coder-body]');hide(incoming,'[data-next-model]');
    }
    container.append(out,incoming,keep,anchors);
    // Copies are visual-only; original data hooks remain unique for export and checks.
    for(const e of [container,...container.querySelectorAll('*')])for(const a of [...e.attributes])if(a.name.startsWith('data-')||a.name==='id'||a.name==='class')e.removeAttribute(a.name);
    transition.append(container);bridges.push({container,out,incoming,keep,anchors});opacity(container,0);
  }
  function bridge(i,q){
    const b=bridges[i],t=ease((q-.1)/.64),outAlpha=1-ease(q/.3),inAlpha=ease((q-.74)/.26);
    opacity(layout,0);opacity(transition,1);bridges.forEach((x,j)=>opacity(x.container,j===i?1:0));
    opacity(b.out,outAlpha);opacity(b.incoming,inAlpha);
    if(i===0){
      const parentCopy=b.anchors.firstElementChild;move(parentCopy,mix(555,38,t),mix(265,122,t));
      parentCopy.querySelectorAll('text').forEach(e=>opacity(e,1-ease(q/.25)));
      parentCopy.querySelector('circle').setAttribute('r',mix(11,8,t));opacity(parentCopy,1);
    }else if(i===1){
      [...b.anchors.children].slice(0,4).forEach((g,j)=>{
        const dot=g.querySelector('circle'),label=g.querySelector('text'),labelWidth=label.getComputedTextLength()/Number(label.getAttribute('font-size'))*21;
        dot.setAttribute('cx',mix(925,xs[j],t));dot.setAttribute('cy',mix(strategyYs[j],100,t));dot.setAttribute('r',mix(6,8,t));
        label.setAttribute('x',mix(947,xs[j]-labelWidth/2,t));label.setAttribute('y',mix(strategyYs[j]+7,133,t));label.setAttribute('font-size',mix(22,21,t));
      });
      const parentDot=b.anchors.lastElementChild;parentDot.setAttribute('cx',mix(38,578,t));parentDot.setAttribute('cy',mix(122,38,t));parentDot.setAttribute('r',mix(8,11,t));
    }else if(i===3){
      const gather=ease((q-.1)/.55);
      [...b.anchors.children].forEach((e,j)=>{
        const from=rolloutPositions[j],to=gatheredPosition(j);
        e.setAttribute('cx',mix(from[0],to[0],gather));e.setAttribute('cy',mix(from[1],to[1],gather));
        e.setAttribute('r',mix(j<44?4.6:4.7,3.4,gather));
      });
    }
    const next=q>=.45;stage=i+Number(next);title.textContent=phases[stage][0];caption.textContent=phases[stage][1];
    if(i===3&&next)title.textContent='Earlier rollout rewards set the shared reference';
    const textAlpha=next?ease((q-.45)/.3):1-ease(q/.3);
    title.style.opacity=String(textAlpha);title.style.transform=`translateY(${next?(1-textAlpha)*6:-6*(1-textAlpha)}px)`;
    caption.style.opacity=String(next?inAlpha:outAlpha);
  }
  function seek(seconds){
    const time=clamp(seconds/total)*total,i=starts.findLastIndex(v=>time>=v),local=time-starts[i];
    setStage(i,clamp((local-(i===4?1.6:.85))/[1.8,8.8,3.2,3.6,4.6,5.2][i]),true);
    const boundary=starts.findIndex((v,j)=>j>0&&time>=v-(j===4?1.2:.55)&&time<v+(j===4?1.6:.85));
    if(boundary>0)bridge(boundary-1,(time-starts[boundary]+(boundary===4?1.2:.55))/(boundary===4?2.8:1.4));
    else if(i===0)sceneElements.forEach(e=>e.style.opacity=String(ease(local/.65)));
    else if(i===4)showUpdate(local);
    return stage;
  }
  setStage(5);return {setStage,seek,total,phases,starts,trainingTimes,get stage(){return stage;}};
}

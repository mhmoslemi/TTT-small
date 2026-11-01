'use strict';

function createFramework(host,data){
  const C={ink:'#202328',grey:'#66717D',pale:'#D7DEE6',blue:'#466B8A',teal:'#25817C',gold:'#A77526',purple:'#806096',red:'#C7000B'};
  const phases=[
    ['A search for a better mathematical solution',''],
    ['Each strategy uses the parent and every earlier strategy',''],
    ['The coder turns each strategy into testable candidates',''],
    ['The bandit allocates more attempts using pilot results','The bandit uses pilot results to assign extra attempts.'],
    ['The system saves candidates and trains the coding model',''],
    ['The search repeats with both updates',''],
  ];
  const durations=[7,11,9,8,9,7],starts=durations.map((_,i)=>durations.slice(0,i).reduce((a,b)=>a+b,0)),total=durations.reduce((a,b)=>a+b,0);
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
    forestNodes+=dot(x,60,6,r===1?C.red:'#B3BDC9');
    [-80,80].forEach((dx,k)=>{
      const bx=x+dx,ancestor=r===1&&k===1;
      forestEdges+=line(branch([x,60],[bx,155]),ancestor?C.red:C.pale,ancestor?2:1.5);
      forestNodes+=dot(bx,155,5.5,ancestor?C.red:'#CAD3DD');
      [-45,0,45].forEach((dd,j)=>{
        const lx=bx+dd,chosen=ancestor&&j===2;
        forestEdges+=line(branch([bx,155],[lx,265]),chosen?C.red:C.pale,chosen?2:1.5);
        if(!chosen)forestNodes+=dot(lx,265,4.7,'#DDE3E9','#C8D1DB',`data-forest-order="${order++}"`);
      });
    });
  });
  let drawing=group('forest',forestEdges+forestNodes);
  drawing+=group('archiveHelp',text(0,20,'In this Erdős example, lower scores are better.',20,C.grey,'start')+text(578,375,'Each dot is a saved candidate. Each link records where it came from.',23,C.ink)+text(578,415,'A candidate is a mathematical construction produced by a program.',21,C.grey));

  let expansion='';
  xs.forEach((x,i)=>{
    expansion+=`<g data-plan="${i}">${line(branch(parent,[x,100]),C.blue,1.6,`data-grow-plan="${i}" pathLength="1"`)}${dot(x,100,8,C.blue)}${text(x,133,`Strategy ${i+1}`,21,C.blue,'middle','font-weight="600"')}</g>`;
  });
  drawing+=group('plans',expansion);
  let planning=text(810,32,'Same frozen model',20,C.blue)+text(1040,32,'New strategy',20,C.blue);
  for(let i=0;i<4;i++){
    const y=99+i*87;
    let inputs=text(30,y+6,'Task + parent',21,C.red,'start');
    for(let j=0;j<i;j++)inputs+=text(208+j*165,y+6,'+',20,C.grey)+text(290+j*165,y+6,`Strategy ${j+1}`,20,C.blue,'middle',`data-prior-strategy="${j}"`);
    planning+=`<g data-strategy-call="${i}"><g data-call-inputs>${inputs}${line(`M 30 ${y+22} H 680`,C.pale,1)}</g>`;
    planning+=line(`M 695 ${y} H 742 M 735 ${y-5} L 742 ${y} L 735 ${y+5}`,C.blue,1.6,'data-call-in pathLength="1"');
    planning+=`<g data-call-model><rect x="758" y="${y-24}" width="111" height="48" rx="4" fill="#F3F6F8" stroke="${C.blue}" stroke-width="1.3"/>${text(813,y+7,'Strategist',20,C.blue)}</g>`;
    planning+=line(`M 885 ${y} H 957 M 950 ${y-5} L 957 ${y} L 950 ${y+5}`,C.blue,1.6,'data-call-out pathLength="1"');
    planning+=`<g data-call-result>${dot(982,y,6,C.blue)}${text(1000,y+7,`Strategy ${i+1}`,22,C.blue,'start','font-weight="600"')}</g></g>`;
  }
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
    pilots+=text(x,269,'11 trial programs',17,C.teal);
    pilots+=`<g data-pilot-score="${i}">${line(`M ${x-67} 278 H ${x+67}`,C.gold,1.3)}${text(x,309,data.allocations[i].pilot_best_raw_score.toFixed(6),20,C.gold)}</g>`;
  });
  pilots+=text(0,309,'Best score',16,C.gold,'start');
  drawing+=group('pilots',pilots);
  drawing+=group('evaluationHelp',text(578,363,'Each dot is one program',23,C.ink,'middle','font-weight="600"')+dot(365,397,5,C.teal)+text(383,403,'Valid output',19,C.teal,'start')+dot(650,397,5,'#E0E5EB','#CDD5DF')+text(668,403,'Invalid output',19,C.grey,'start')+text(578,450,'The evaluator runs the code, checks its output, and assigns a score.',21,C.ink));

  let followups=text(578,411,'20 extra attempts to allocate',21,C.purple,'middle','data-pool-label="true"');
  const extraPositions=[];
  for(let j=0;j<20;j++){
    const lane=j<17?2:3,index=j<17?j:j-17,n=lane===2?17:3,row=Math.floor(index/9),cols=Math.min(9,n-row*9);
    const target=[xs[lane]+(index%9-(cols-1)/2)*13,353+row*14];extraPositions.push(target);
    followups+=dot(578+(j%10-4.5)*13,360+Math.floor(j/10)*14,4.7,C.purple,C.purple,`data-extra="${j}"`);
  }
  xs.forEach((x,i)=>{followups+=text(x,399,`${data.allocations[i].followup_count} extra attempts`,19,i>1?C.purple:C.grey,'middle',`data-extra-count="${i}" font-weight="${i>1?600:400}"`);});
  followups+=`<g data-child-score="true">${text(590,450,'Best new score',20,C.red,'end')}${dot(720,423,5,C.red)}${text(720,455,data.allocationFinalBest.toFixed(6),26,C.red,'middle','font-weight="600"')}${text(1005,454,data.allocations[3].final_best_raw_score.toFixed(6),18,C.gold)}</g>`;
  drawing+=group('followups',followups);

  let learning=text(180,38,'Saved candidates (archive)',25,C.ink,'middle','font-weight="600"');
  learning+=line(branch([180,105],[85,245]))+dot(85,245,5,'#CDD6DF');
  learning+=line(branch([180,105],[320,245]),C.red,2)+dot(320,245,11,C.red);
  learning+=text(320,286,'New candidate (child)',20,C.red,'middle','font-weight="600"')+text(320,317,data.allocationFinalBest.toFixed(6),23,C.red);
  learning+=text(900,45,'SPO² training',30,C.red,'middle','font-weight="600"')+text(900,107,'Coding model',24,C.ink,'middle','font-weight="600"');
  learning+=dot(900,201,65,'#FFFFFF',C.red,'data-policy-ring="true"')+text(900,209,'Before',24,C.ink,'middle','data-policy-theta="true"');
  learning+=text(900,298,'Learns from evaluated programs',22,C.red,'middle','font-weight="600"')+text(900,339,'One update after the full batch',19,C.grey);
  learning+=text(620,92,'Results from all parents',17,C.grey);
  learning+=text(900,411,'Planning model stays unchanged',19,C.blue);
  drawing+=group('learning',learning);
  let outcomes='';
  [...pilotPositions,...extraPositions].forEach(([x,y],i)=>{outcomes+=dot(x,y,3.4,i<44&&!(i%11<data.allocations[Math.floor(i/11)].pilot_valid_count)?'#BEC7D1':C.teal,undefined,`data-outcome="${i}"`);});
  drawing+=group('outcomes',outcomes);

  let next=line(branch([407,197],[480,280]),C.red,2);
  next+=text(510,261,'Saved child',19,C.red,'start','font-weight="600"')+text(510,289,data.allocationFinalBest.toFixed(6),22,C.red,'start');
  [410,480,550].forEach((x,i)=>{next+=line(branch([480,280],[x,397]),C.teal,1.5,`data-new-branch="${i}" pathLength="1"`)+dot(x,397,6,'#FFFFFF',C.teal,`data-new-node="${i}"`);});
  next+=dot(480,280,10,C.red)+text(480,437,'New attempts',21,C.teal);
  next+=text(950,196,'Coding model',24,C.ink,'middle','font-weight="600"')+text(950,251,'Updated',38,C.red);
  next+=line(curve([890,269],[558,397],-32),C.red,2,'data-next-flow="true" pathLength="1"')+line('M 570 392 L 558 397 L 570 402',C.red,2,'data-next-head="true"');
  next+=text(1000,332,'Generates the next programs',18,C.red,'middle','data-next-caption="true"')+text(0,24,'Saved candidates',23,C.ink,'start','font-weight="600"');
  drawing+=group('next',next);
  drawing+=group('parent',dot(0,0,11,C.red)+text(0,33,'Starting candidate (parent)',20,C.ink,'middle','data-parent-name="true" font-weight="600"')+text(0,59,`Score: ${data.lineage[0].score.toFixed(6)}`,18,C.grey,'middle','data-parent-score="true"'));

  host.innerHTML=`<svg class="data-figure framework-figure" viewBox="0 0 1156 480" role="img" aria-label="Animated search archive and one parent expansion. Each strategist call receives the task, parent and every earlier strategy for that parent. Four strategies receive equal pilots, then 20 follow-ups are allocated as zero, zero, 17 and 3. A retained child joins the archive, all parents contribute to SPO squared training, and the next step uses the updated coder."><title>Hierarchical scientific discovery</title>${drawing}</svg>`;
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
    sceneElements.forEach(e=>e.style.opacity='1');
    title.textContent=phases[stage][0];count.textContent='';caption.textContent=phases[stage][1];
    Object.values(groups).forEach(g=>opacity(g,0));
    groups.forest.setAttribute('transform',stage===5?'translate(35 20) scale(.67)':'translate(0 0)');
    const root=stage===0?selected:stage===4?[180,105]:stage===5?[407,197]:parent;
    move(groups.parent,...root);opacity(groups.parent,stage===1?0:1);
    const name=host.querySelector('[data-parent-name]'),value=host.querySelector('[data-parent-score]');
    name.textContent=stage===0?'Starting candidate (parent)':'Parent';name.setAttribute('x',stage===0?0:-26);name.setAttribute('y',stage===0?33:6);name.setAttribute('text-anchor',stage===0?'middle':'end');
    value.setAttribute('x',stage===0?0:26);value.setAttribute('y',stage===0?59:6);value.setAttribute('text-anchor',stage===0?'middle':'start');
    opacity(name,stage===5?0:1);opacity(value,stage===5?0:1);
    if(stage===0){opacity(groups.forest,ease(p*3));opacity(groups.archiveHelp,ease((p-.15)*3));}
    if(stage===1)opacity(groups.planning,1);
    host.querySelectorAll('[data-strategy-call]').forEach(g=>{
      const a=p*4-Number(g.dataset.strategyCall);
      opacity(g,ease(a*6));
      opacity(g.querySelector('[data-call-inputs]'),ease(a*4));
      opacity(g.querySelector('[data-call-model]'),ease((a-.24)*5));
      opacity(g.querySelector('[data-call-result]'),ease((a-.65)/.25));
      for(const [selector,start] of [['[data-call-in]',.15],['[data-call-out]',.48]]){
        const edge=g.querySelector(selector);edge.setAttribute('stroke-dasharray','1');edge.setAttribute('stroke-dashoffset',String(1-ease((a-start)/.22)));
      }
    });
    if(stage===2||stage===3){opacity(groups.plans,1);opacity(groups.pilots,1);}
    if(stage===2)opacity(groups.evaluationHelp,ease((p-.45)*3));
    host.querySelectorAll('[data-plan]').forEach(g=>{const k=Number(g.dataset.plan);opacity(g,stage===1?ease((p-.2)*3-k*.24):1);});
    host.querySelectorAll('[data-grow-plan]').forEach(e=>{e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',stage===1?String(1-ease((p-.2)*3-Number(e.dataset.growPlan)*.24)):'0');});
    host.querySelectorAll('[data-coder]').forEach(e=>opacity(e,stage===2?ease(p/.18):1));
    host.querySelectorAll('[data-coder-output]').forEach(e=>opacity(e,stage===2?ease((p-.18)/.12):1));
    host.querySelectorAll('[data-pilot]').forEach(e=>{
      const i=Number(e.dataset.pilot),k=Math.floor(i/11),j=i%11,[x,y]=pilotPositions[i],a=stage===2?ease((p-.3)*3-j*.06):1;
      e.setAttribute('cx',mix(xs[k],x,a));e.setAttribute('cy',mix(228,y,a));opacity(e,a);
      const evaluated=stage>2||p>.58+j*.025,valid=e.dataset.pilotValid==='true';
      e.setAttribute('fill',evaluated&&!valid?'#E0E5EB':C.teal);e.setAttribute('stroke',evaluated&&!valid?'#CDD5DF':C.teal);
    });
    host.querySelectorAll('[data-pilot-score]').forEach(e=>opacity(e,stage===2?ease((p-.65)*4):1));
    if(stage===3)opacity(groups.followups,1);
    host.querySelectorAll('[data-extra]').forEach(e=>{
      const i=Number(e.dataset.extra),a=stage===3?ease((p-.18)*2.4-i*.018):1,[x,y]=extraPositions[i];
      e.setAttribute('cx',mix(578+(i%10-4.5)*13,x,a));e.setAttribute('cy',mix(360+Math.floor(i/10)*14,y,a)-Math.sin(a*Math.PI)*45);opacity(e,stage===3?ease(p*7):1);
    });
    opacity(host.querySelector('[data-pool-label]'),stage===3?1-ease((p-.15)*4):0);
    host.querySelectorAll('[data-extra-count]').forEach(e=>opacity(e,stage===3?ease((p-.55)*4):1));
    opacity(host.querySelector('[data-child-score]'),stage===3?ease((p-.83)*6):1);
    if(stage===4){
      opacity(groups.learning,1);opacity(groups.outcomes,ease((p-.15)*6));
    }
    host.querySelectorAll('[data-outcome]').forEach(e=>{
      const i=Number(e.dataset.outcome),from=[620+(i%8-3.5)*9,125+Math.floor(i/8)*9],a=ease((p-.15)*1.8-i*.006);
      e.setAttribute('cx',mix(from[0],900,a));e.setAttribute('cy',mix(from[1],201,a)-Math.sin(a*Math.PI)*8);opacity(e,stage===4?(1-ease((a-.85)/.15))*.8:0);
    });
    host.querySelector('[data-policy-theta]').textContent=stage===4&&p<.94?'Before':'Updated';
    host.querySelector('[data-policy-ring]').setAttribute('r',stage===4&&p>.92?String(65+4*Math.sin((p-.92)/.08*Math.PI)):65);
    if(stage===5){opacity(groups.forest,1);opacity(groups.next,1);}
    host.querySelectorAll('[data-new-branch]').forEach(e=>{e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',String(1-(stage===5?ease((p-.1)*3.2-Number(e.dataset.newBranch)*.16):0)));});
    host.querySelectorAll('[data-new-node]').forEach(e=>opacity(e,stage===5?ease((p-.25)*4-Number(e.dataset.newNode)*.16):0));
    const flow=stage===5?ease((p-.62)/.32):0,returnPath=host.querySelector('[data-next-flow]');
    returnPath.setAttribute('stroke-dasharray','1');returnPath.setAttribute('stroke-dashoffset',String(1-flow));opacity(returnPath,flow>0?1:0);
    opacity(host.querySelector('[data-next-head]'),stage===5?ease((p-.94)/.06):0);
    opacity(host.querySelector('[data-next-caption]'),ease(flow*3));
    return stage;
  }
  function seek(seconds){
    const time=clamp(seconds/total)*total,i=starts.findLastIndex(v=>time>=v),local=time-starts[i];
    setStage(i,clamp((local-.35)/[1.8,7.2,3.2,3.6,3.4,3.5][i]),true);
    // Change layouts only while invisible, then hold the completed scene for reading.
    const alpha=Math.min(ease(local/.35),i===5?1:ease((durations[i]-local)/.25));
    sceneElements.forEach(e=>e.style.opacity=String(alpha));
    return i;
  }
  setStage(5);return {setStage,seek,total,phases,starts,get stage(){return stage;}};
}

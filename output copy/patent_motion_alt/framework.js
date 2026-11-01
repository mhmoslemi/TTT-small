'use strict';

function createFramework(host,data){
  const C={ink:'#202328',grey:'#66717D',pale:'#D7DEE6',blue:'#466B8A',teal:'#25817C',gold:'#A77526',purple:'#806096',red:'#C7000B'};
  const phases=[
    ['Search begins with a scored archive','Select P parents. Each carries a program, a construction, and its evaluated score.'],
    ['One parent opens several directions','A frozen strategist proposes sequential plans. Each new plan can see the earlier ones.'],
    ['Equal pilots test each strategy','The shared coder generates 11 programs per plan. Evaluation returns raw scores and marks invalid outputs.'],
    ['Evaluation directs the remaining budget','The bandit allocates 20 follow-ups: 0, 0, 17, and 3. The best child reaches C₅ = 0.380979914.'],
    ['Archive retention and SPO² training','After generation ends, ready results train the shared coder while CPU evaluation finishes. Apply one final update.'],
    ['The next step uses both updates','The archive contains the retained child, and the shared coder uses its updated parameters.'],
  ];
  const durations=[3.5,4,4.5,6,5.5,5],starts=durations.map((_,i)=>durations.slice(0,i).reduce((a,b)=>a+b,0)),total=durations.reduce((a,b)=>a+b,0);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,s,size=20,color=C.ink,anchor='middle',extra='')=>`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" style="fill:${color}" ${extra}>${esc(s)}</text>`;
  const dot=(x,y,r,fill,stroke=fill,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5" ${extra}/>`;
  const line=(d,color=C.pale,width=1.5,extra='')=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const group=(id,body)=>`<g data-scene="${id}">${body}</g>`;
  const curve=(a,b,bend=0)=>{
    const points=Array.from({length:17},(_,i)=>{const t=i/16,u=1-t;return [u*a[0]+t*b[0],u*a[1]+t*b[1]-4*t*u*bend];});
    return points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ');
  };
  const xs=[150,435,720,1005],selected=[555,265],parent=[578,38];
  let forest='',order=0;
  [140,430,730,1020].forEach((x,r)=>{
    forest+=dot(x,60,6,'#B3BDC9');
    [-80,80].forEach((dx,k)=>{
      const bx=x+dx;forest+=line(curve([x,66],[bx,155],14))+dot(bx,155,5.5,'#CAD3DD');
      [-45,0,45].forEach((dd,j)=>{
        const lx=bx+dd;forest+=line(curve([bx,160],[lx,265],13));
        if(!(r===1&&k===1&&j===2))forest+=dot(lx,265,4.7,'#DDE3E9','#C8D1DB',`data-forest-order="${order++}"`);
      });
    });
  });
  forest+=line('M 430 60 L 510 155 L 555 265',C.red,2,'data-selection-path="true"');
  forest+=dot(105,265,11,'none',C.blue)+dot(985,265,11,'none',C.blue);
  let drawing=group('forest',forest);

  let expansion='';
  xs.forEach((x,i)=>{
    expansion+=`<g data-plan="${i}">${line(curve([578,52],[x,129],20),C.blue,1.6,`data-grow-plan="${i}" pathLength="1"`)}${dot(x,135,8,C.blue)}${text(x,172,`Strategy ${i+1}`,21,C.blue,'middle','font-weight="600"')}</g>`;
  });
  expansion+=text(578,105,'Frozen strategist',17,C.blue);
  drawing+=group('plans',expansion);

  let pilots=text(0,210,'Shared coder',18,C.teal,'start','font-weight="600"');
  const pilotPositions=[];
  xs.forEach((x,i)=>{
    for(let j=0;j<11;j++){
      const px=x+(j-5)*12,py=230;pilotPositions.push([px,py]);
      pilots+=dot(px,py,4.6,C.teal,C.teal,`data-pilot="${i*11+j}" data-pilot-valid="${j<data.allocations[i].pilot_valid_count}"`);
    }
    pilots+=text(x,259,'11 pilot programs',17,C.teal);
    pilots+=`<g data-pilot-score="${i}">${line(`M ${x-67} 278 H ${x+67}`,C.gold,1.3)}${text(x,309,data.allocations[i].pilot_best_raw_score.toFixed(9),20,C.gold)}</g>`;
  });
  pilots+=text(0,309,'Best C₅',16,C.gold,'start');
  drawing+=group('pilots',pilots);

  let followups=text(578,411,'20 remaining programs',21,C.purple,'middle','data-pool-label="true"');
  const extraPositions=[];
  for(let j=0;j<20;j++){
    const lane=j<17?2:3,index=j<17?j:j-17,n=lane===2?17:3,row=Math.floor(index/9),cols=Math.min(9,n-row*9);
    const target=[xs[lane]+(index%9-(cols-1)/2)*13,353+row*14];extraPositions.push(target);
    followups+=dot(578+(j%10-4.5)*13,360+Math.floor(j/10)*14,4.7,C.purple,C.purple,`data-extra="${j}"`);
  }
  xs.forEach((x,i)=>{followups+=text(x,399,`${data.allocations[i].followup_count} follow-ups`,19,i>1?C.purple:C.grey,'middle',`data-extra-count="${i}" font-weight="${i>1?600:400}"`);});
  followups+=`<g data-child-score="true">${dot(720,423,5,C.red)}${text(720,455,data.allocationFinalBest.toFixed(9),26,C.red,'middle','font-weight="600"')}${text(1005,454,data.allocations[3].final_best_raw_score.toFixed(9),18,C.gold)}</g>`;
  drawing+=group('followups',followups);

  let learning=text(180,38,'Archive',25,C.ink,'middle','font-weight="600"');
  learning+=line(curve([180,119],[85,245],22))+dot(85,245,5,'#CDD6DF');
  learning+=line(curve([180,119],[320,245],22),C.red,2)+dot(320,245,11,C.red);
  learning+=text(320,286,'Retained child',20,C.red,'middle','font-weight="600"')+text(320,317,data.allocationFinalBest.toFixed(9),23,C.red);
  learning+=text(900,45,'SPO²',40,C.red,'middle','font-weight="600"')+text(900,107,'Shared coder',24,C.ink,'middle','font-weight="600"');
  learning+=dot(900,201,57,'#FFFFFF',C.red,'data-policy-ring="true"')+text(900,218,'θₜ',49,C.ink,'middle','data-policy-theta="true"');
  learning+=text(900,298,'One optimizer update',22,C.red,'middle','font-weight="600"')+text(900,334,'A fixed historical reference',17,C.grey)+text(900,356,'within the step',17,C.grey);
  learning+=text(670,92,'All parents contribute',18,C.grey);
  learning+=text(900,411,'Strategist stays frozen',19,C.blue);
  drawing+=group('learning',learning);
  let outcomes='';
  [...pilotPositions,...extraPositions].forEach(([x,y],i)=>{outcomes+=dot(x,y,3.4,i<44&&!(i%11<data.allocations[Math.floor(i/11)].pilot_valid_count)?'#BEC7D1':C.teal,undefined,`data-outcome="${i}"`);});
  drawing+=group('outcomes',outcomes);

  let next=line(curve([407,197],[480,280],-15),C.red,2)+dot(480,280,10,C.red);
  next+=text(510,261,'Retained child',19,C.red,'start','font-weight="600"')+text(510,289,data.allocationFinalBest.toFixed(9),22,C.red,'start');
  [410,480,550].forEach((x,i)=>{next+=line(curve([480,291],[x,397],15),C.teal,1.5,`data-new-branch="${i}" pathLength="1"`)+dot(x,397,6,'#FFFFFF',C.teal,`data-new-node="${i}"`);});
  next+=text(480,442,'Next expansion',21,C.teal);
  next+=text(950,115,'Frozen strategist',20,C.blue)+text(950,196,'Shared coder',24,C.ink,'middle','font-weight="600"')+text(950,259,'θₜ₊₁',51,C.red);
  next+=line(curve([890,269],[560,397],-55),C.red,2,'data-next-flow="true" pathLength="1"')+line('M 572 389 L 560 397 L 574 398',C.red,2);
  next+=text(950,332,'Updated policy',19,C.red)+text(0,24,'Expanded archive',23,C.ink,'start','font-weight="600"');
  drawing+=group('next',next);
  drawing+=group('parent',dot(0,0,11,C.red)+dot(0,0,20,'none',C.red,'data-parent-halo="true"')+text(0,33,'Selected parent',20,C.ink,'middle','data-parent-name="true" font-weight="600"')+text(0,59,data.lineage[0].score.toFixed(9),18,C.grey,'middle','data-parent-score="true"'));

  host.innerHTML=`<svg class="data-figure framework-figure" viewBox="0 0 1156 480" role="img" aria-label="Animated search archive and one parent expansion. Four strategies receive equal pilots, then 20 follow-ups are allocated as zero, zero, 17 and 3. A retained child joins the archive, all parents contribute to SPO squared training, and the next step uses the updated coder."><title>Hierarchical scientific discovery</title>${drawing}</svg>`;
  const groups=Object.fromEntries([...host.querySelectorAll('[data-scene]')].map(g=>[g.dataset.scene,g]));
  const title=host.closest('.slide')?.querySelector('.framework-stage-title')||document.getElementById('framework-stage-title');
  const count=host.closest('.slide')?.querySelector('.framework-stage-count')||document.getElementById('framework-stage-count');
  const caption=host.closest('.slide')?.querySelector('.framework-caption')||document.getElementById('framework-caption');
  const clamp=t=>Math.max(0,Math.min(1,t)),ease=t=>{t=clamp(t);return t*t*(3-2*t);},mix=(a,b,t)=>a+(b-a)*t;
  const opacity=(e,v)=>{e.style.opacity=String(clamp(v));e.style.visibility=v>0?'visible':'hidden';};
  const move=(e,x,y)=>e.setAttribute('transform',`translate(${x} ${y})`);
  let stage=5;
  function setStage(s,progress=1,moving=false){
    stage=Math.max(0,Math.min(5,s));const p=clamp(progress),t=ease(p*1.6);
    title.textContent=phases[stage][0];count.textContent=`${String(stage+1).padStart(2,'0')} / 06`;caption.textContent=phases[stage][1];
    Object.values(groups).forEach(g=>opacity(g,0));
    groups.forest.setAttribute('transform',stage===5?'translate(35 20) scale(.67)':'translate(0 0)');
    const root=stage===0?selected:stage===1?[mix(selected[0],parent[0],t),mix(selected[1],parent[1],t)]:stage===4?[mix(578,180,t),mix(38,105,t)]:stage===5?[mix(180,407,t),mix(105,197,t)]:parent;
    move(groups.parent,...root);opacity(groups.parent,1);
    const name=host.querySelector('[data-parent-name]'),value=host.querySelector('[data-parent-score]');
    name.textContent=stage===0?'Selected parent':'Parent p';name.setAttribute('x',stage===0?0:-26);name.setAttribute('y',stage===0?33:6);name.setAttribute('text-anchor',stage===0?'middle':'end');
    value.setAttribute('x',stage===0?0:26);value.setAttribute('y',stage===0?59:6);value.setAttribute('text-anchor',stage===0?'middle':'start');
    opacity(name,stage===5?0:1);opacity(value,stage===5?0:1);
    host.querySelector('[data-parent-halo]').setAttribute('r',moving&&stage===0?String(20+3*Math.sin(p*Math.PI*2)):20);
    if(stage===0){opacity(groups.forest,ease(p*3));}
    if(stage===1){opacity(groups.forest,1-t);opacity(groups.plans,ease((p-.2)*2));}
    if(stage===2||stage===3){opacity(groups.plans,1);opacity(groups.pilots,1);}
    host.querySelectorAll('[data-plan]').forEach(g=>{const k=Number(g.dataset.plan);opacity(g,stage===1?ease((p-.2)*3-k*.24):1);});
    host.querySelectorAll('[data-grow-plan]').forEach(e=>{e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',stage===1?String(1-ease((p-.2)*3-Number(e.dataset.growPlan)*.24)):'0');});
    host.querySelectorAll('[data-pilot]').forEach(e=>{
      const i=Number(e.dataset.pilot),k=Math.floor(i/11),j=i%11,[x,y]=pilotPositions[i],a=stage===2?ease(p*2.8-j*.075):1;
      e.setAttribute('cx',mix(xs[k],x,a));e.setAttribute('cy',mix(188,y,a));opacity(e,a);
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
      opacity(groups.plans,1-t);opacity(groups.pilots,1-t);opacity(groups.followups,1-t);opacity(groups.learning,t);opacity(groups.outcomes,ease((p-.15)*6));
    }
    host.querySelectorAll('[data-outcome]').forEach(e=>{
      const i=Number(e.dataset.outcome),from=[...pilotPositions,...extraPositions][i],a=ease((p-.15)*1.8-i*.006);
      e.setAttribute('cx',mix(from[0],900,a));e.setAttribute('cy',mix(from[1],201,a)-Math.sin(a*Math.PI)*(35+i%5*12));opacity(e,stage===4?(1-ease((a-.85)/.15))*.8:0);
    });
    host.querySelector('[data-policy-theta]').textContent=stage===4&&p<.94?'θₜ':'θₜ₊₁';
    host.querySelector('[data-policy-ring]').setAttribute('r',stage===4&&p>.92?String(57+4*Math.sin((p-.92)/.08*Math.PI)):57);
    if(stage===5){opacity(groups.learning,1-t);opacity(groups.forest,t);opacity(groups.next,t);}
    host.querySelectorAll('[data-new-branch]').forEach(e=>{e.setAttribute('stroke-dasharray','1');e.setAttribute('stroke-dashoffset',String(1-(stage===5?ease((p-.35)*2.4-Number(e.dataset.newBranch)*.16):0)));});
    host.querySelectorAll('[data-new-node]').forEach(e=>opacity(e,stage===5?ease((p-.55)*3-Number(e.dataset.newNode)*.16):0));
    const returnPath=host.querySelector('[data-next-flow]');returnPath.setAttribute('stroke-dasharray','1');returnPath.setAttribute('stroke-dashoffset',String(1-(stage===5?ease(p*2):0)));
    return stage;
  }
  function seek(seconds){const time=clamp(seconds/total)*total,i=starts.findLastIndex(v=>time>=v);return setStage(i,Math.min(1,(time-starts[i])/durations[i]),true);}
  setStage(5);return {setStage,seek,total,phases,starts,get stage(){return stage;}};
}

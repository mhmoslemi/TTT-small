export function extractEditableText() {
  const slide = document.querySelector('.slide.active');
  const lines = [], owners = new Map();
  const color = value => {
    const rgb = value.match(/[\d.]+/g);
    if (value.startsWith('rgb') && rgb) return rgb.slice(0, 3).map(v => Math.round(Number(v)).toString(16).padStart(2, '0')).join('').toUpperCase();
    return value.replace('#', '').toUpperCase();
  };
  const visible = element => {
    for (let e = element; e && e !== slide.parentElement; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0) return false;
    }
    return true;
  };
  const styleOf = (element, scale = 1, svg = false) => {
    const s = getComputedStyle(element);
    return {font:s.fontFamily.split(',')[0].replaceAll('"','').trim(),size:parseFloat(s.fontSize)*scale,
      bold:Number(s.fontWeight) >= 600,italic:s.fontStyle === 'italic',color:color(svg?s.fill:s.color),
      spacing:s.letterSpacing === 'normal'?0:parseFloat(s.letterSpacing)*scale};
  };
  const walker = document.createTreeWalker(slide, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const parent = node.parentElement;
    if (parent.closest('svg,math,.source-notes,.build-caption') || !visible(parent)) continue;
    const owner = parent.closest('h1,h2,p,li,td,th,.takeaway,.number,.review-label,.thankyou,.score-large,.legend>span,footer>span') || parent;
    const groups = owners.get(owner) || [];
    owners.set(owner, groups);
    const style = styleOf(parent), range = document.createRange();
    let offset = 0;
    for (let char of node.textContent) {
      range.setStart(node, offset); offset += char.length; range.setEnd(node, offset);
      const rect = range.getBoundingClientRect();
      if (rect.width < .01 || rect.height < .01) continue;
      if (getComputedStyle(parent).textTransform === 'uppercase') char = char.toUpperCase();
      let line = groups.find(l => Math.abs(l.y-rect.y)<.8 && Math.abs(l.height-rect.height)<.8 && rect.left<=l.x+l.width+.8 && rect.right>=l.x);
      if (!line) {line={x:rect.x,y:rect.y,width:rect.width,height:rect.height,runs:[],rotation:0,kind:owner.tagName.toLowerCase()};groups.push(line);}
      const last = line.runs.at(-1);
      if (last && JSON.stringify(last.style) === JSON.stringify(style)) last.text += char;
      else line.runs.push({text:char,style});
      const right = Math.max(line.x+line.width, rect.right);
      line.x = Math.min(line.x,rect.x); line.width = right-line.x;
    }
  }
  for (const [owner,groups] of owners) {
    lines.push(...groups);
    if (owner.tagName !== 'LI' || !groups.length) continue;
    const first=groups[0],style=styleOf(owner),type=getComputedStyle(owner).listStyleType;
    if (type === 'none') continue;
    const ordered=owner.parentElement.tagName === 'OL';
    const text=ordered?`${[...owner.parentElement.children].indexOf(owner)+Number(owner.parentElement.start||1)}.`:type==='circle'?'◦':'•';
    lines.push({x:first.x-style.size*(ordered?1.45:1.1),y:first.y,width:style.size*1.2,height:first.height,
      runs:[{text,style}],rotation:0,kind:'marker'});
  }
  for (const text of slide.querySelectorAll('svg text')) {
    if (!visible(text) || !text.textContent.trim()) continue;
    const m = text.getScreenCTM(), b = text.getBBox(), scale = Math.hypot(m.a,m.b);
    const center = new DOMPoint(b.x+b.width/2,b.y+b.height/2).matrixTransform(m);
    const height = b.height*scale, width = b.width*scale;
    lines.push({x:center.x-width/2,y:center.y-height/2,width,height,rotation:Math.atan2(m.b,m.a)*180/Math.PI,kind:'svg',
      runs:[{text:text.textContent,style:styleOf(text,scale,true)}]});
  }
  return lines.filter(line=>line.runs.some(r=>r.text.trim())).sort((x,y)=>x.y-y.y||x.x-y.x);
}

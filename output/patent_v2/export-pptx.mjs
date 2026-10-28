import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const out = path.dirname(fileURLToPath(import.meta.url));
const build = process.argv[2];
if (!build || !fs.existsSync(path.join(build, 'editable-pptx.json'))) throw new Error('Run render.mjs with --editable-pptx-assets and pass its private build directory');
const {slides} = JSON.parse(fs.readFileSync(path.join(build, 'editable-pptx.json'), 'utf8'));
assert.equal(slides.length, 30);
for (const [page,count] of [[7,6],[9,4],[10,4]]) assert.equal(slides.filter(s => s.page === page).length,count);
const notesMasterId = `rId${slides.length + 2}`;

// shortcut: equations and plot geometry remain artwork; slide text and figure labels are native text boxes.
const pkg = fs.mkdtempSync(path.join(build, 'pptx-package-'));
const ns = 'http://schemas.openxmlformats.org/';
const a = `${ns}drawingml/2006/main`, p = `${ns}presentationml/2006/main`, r = `${ns}officeDocument/2006/relationships`;
const decl = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>';
const xml = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const files = new Set();
function write(name, content) {
  const target = path.join(pkg, name); fs.mkdirSync(path.dirname(target), {recursive:true});
  fs.writeFileSync(target, decl + content); files.add(name);
}
function media(name, source) {
  const target = path.join(pkg, name); fs.mkdirSync(path.dirname(target), {recursive:true});
  fs.copyFileSync(source, target); files.add(name);
}
function rels(name, rows) {
  for (const row of rows) {
    const ownerDir = path.posix.dirname(name).replace(/(?:^|\/)\_rels$/, '');
    const target = path.posix.normalize(path.posix.join(ownerDir, row.target));
    assert(!target.startsWith('../'), `Invalid relationship: ${target}`);
  }
  write(name, `<Relationships xmlns="${ns}package/2006/relationships">${rows.map(x=>`<Relationship Id="${x.id}" Type="${x.type.startsWith('http')?x.type:r+'/'+x.type}" Target="${xml(x.target)}"/>`).join('')}</Relationships>`);
}
const group = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>';
const colorMap = '<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>';
const header = `xmlns:a="${a}" xmlns:r="${r}" xmlns:p="${p}"`;
const emu = n => Math.round(n * 9525);
function picture(id, name, rid, x, y, width, height) {
  return `<p:pic><p:nvPicPr><p:cNvPr id="${id}" name="${xml(name)}" descr="${xml(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="${rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="${emu(x)}" y="${emu(y)}"/><a:ext cx="${emu(width)}" cy="${emu(height)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>`;
}
function textBox(id, item) {
  assert(item.runs.length && item.runs.every(r=>r.style.size>0 && /^[0-9A-F]{6}$/.test(r.style.color)));
  const rotation=((item.rotation||0)+360)%360, vertical=Math.abs(rotation-270)<.1;
  const width=vertical?item.height+2:item.kind==='h1'?1218-item.x:item.width+(item.kind==='svg'?5:Math.max(5,item.runs[0].style.size*.7));
  const height=vertical?item.width+5:item.height+2;
  const left=vertical?item.x+item.width/2-width/2:item.x, top=vertical?item.y+item.height/2-height/2:item.y-1;
  const runs=item.runs.map(({text,style:s})=>`<a:r><a:rPr lang="en-US" sz="${Math.round(s.size*75)}" b="${s.bold?1:0}" i="${s.italic?1:0}" spc="${Math.round(s.spacing*75)}"><a:solidFill><a:srgbClr val="${s.color}"/></a:solidFill><a:latin typeface="${xml(s.font)}"/></a:rPr><a:t xml:space="preserve">${xml(text)}</a:t></a:r>`).join('');
  return `<p:sp><p:nvSpPr><p:cNvPr id="${id}" name="${xml(item.runs.map(r=>r.text).join('').slice(0,70))}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm rot="${vertical?0:Math.round(rotation*60000)}"><a:off x="${emu(left)}" y="${emu(top)}"/><a:ext cx="${emu(width)}" cy="${emu(height)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr vert="${vertical?'vert270':'horz'}" wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr" vertOverflow="overflow" horzOverflow="overflow"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr algn="${vertical?'ctr':'l'}"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>${runs}<a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>`;
}
const overrides = [
  ['/ppt/presentation.xml','presentationml.presentation.main'], ['/ppt/slideMasters/slideMaster1.xml','presentationml.slideMaster'],
  ['/ppt/slideLayouts/slideLayout1.xml','presentationml.slideLayout'], ['/ppt/notesMasters/notesMaster1.xml','presentationml.notesMaster'],
  ['/ppt/theme/theme1.xml','theme'], ['/docProps/core.xml','core-properties'], ['/docProps/app.xml','extended-properties'],
];
for (let i=1; i<=slides.length; i++) overrides.push([`/ppt/slides/slide${i}.xml`,'presentationml.slide'],[`/ppt/notesSlides/notesSlide${i}.xml`,'presentationml.notesSlide']);
write('[Content_Types].xml', `<Types xmlns="${ns}package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>${overrides.map(([name,type])=>`<Override PartName="${name}" ContentType="application/vnd.openxmlformats-${type==='core-properties'?'package.':'officedocument.'}${type}+xml"/>`).join('')}</Types>`);
rels('_rels/.rels', [{id:'rId1',type:'officeDocument',target:'ppt/presentation.xml'}, {id:'rId2',type:`${ns}package/2006/relationships/metadata/core-properties`,target:'docProps/core.xml'}, {id:'rId3',type:'extended-properties',target:'docProps/app.xml'}]);
write('docProps/core.xml', '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Hierarchical Bandit-Guided Scientific Discovery</dc:title><dc:description>Editable slide text and figure labels, static sequential builds, original page labels, and speaker notes.</dc:description></cp:coreProperties>');
write('docProps/app.xml', `<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Scientific Discovery Presentation</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>${slides.length}</Slides><Notes>${slides.length}</Notes><HiddenSlides>0</HiddenSlides></Properties>`);
write('ppt/presentation.xml', `<p:presentation ${header}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="${notesMasterId}"/></p:notesMasterIdLst><p:sldIdLst>${slides.map((_,i)=>`<p:sldId id="${256+i}" r:id="rId${i+2}"/>`).join('')}</p:sldIdLst><p:sldSz cx="${emu(1280)}" cy="${emu(720)}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle></p:presentation>`);
rels('ppt/_rels/presentation.xml.rels', [{id:'rId1',type:'slideMaster',target:'slideMasters/slideMaster1.xml'},...slides.map((_,i)=>({id:`rId${i+2}`,type:'slide',target:`slides/slide${i+1}.xml`})),{id:notesMasterId,type:'notesMaster',target:'notesMasters/notesMaster1.xml'}]);
write('ppt/slideMasters/slideMaster1.xml', `<p:sldMaster ${header}><p:cSld name="PDF-matched master"><p:spTree>${group}</p:spTree></p:cSld>${colorMap}<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>`);
rels('ppt/slideMasters/_rels/slideMaster1.xml.rels', [{id:'rId1',type:'slideLayout',target:'../slideLayouts/slideLayout1.xml'},{id:'rId2',type:'theme',target:'../theme/theme1.xml'}]);
write('ppt/slideLayouts/slideLayout1.xml', `<p:sldLayout ${header} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>${group}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>`);
rels('ppt/slideLayouts/_rels/slideLayout1.xml.rels', [{id:'rId1',type:'slideMaster',target:'../slideMasters/slideMaster1.xml'}]);
const themeColors = {dk1:'202328',lt1:'FFFFFF',dk2:'606873',lt2:'EEF3F8',accent1:'C7000B',accent2:'466B8A',accent3:'25817C',accent4:'A77526',accent5:'806096',accent6:'BD343A',hlink:'0563C1',folHlink:'954F72'};
const fill = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>';
const font = '<a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/>';
write('ppt/theme/theme1.xml', `<a:theme xmlns:a="${a}" name="Huawei"><a:themeElements><a:clrScheme name="Huawei">${Object.entries(themeColors).map(([key,value])=>`<a:${key}><a:srgbClr val="${value}"/></a:${key}>`).join('')}</a:clrScheme><a:fontScheme name="Arial"><a:majorFont>${font}</a:majorFont><a:minorFont>${font}</a:minorFont></a:fontScheme><a:fmtScheme name="Simple"><a:fillStyleLst>${fill.repeat(3)}</a:fillStyleLst><a:lnStyleLst>${[6350,12700,19050].map(w=>`<a:ln w="${w}" cap="flat" cmpd="sng" algn="ctr">${fill}<a:prstDash val="solid"/></a:ln>`).join('')}</a:lnStyleLst><a:effectStyleLst>${'<a:effectStyle><a:effectLst/></a:effectStyle>'.repeat(3)}</a:effectStyleLst><a:bgFillStyleLst>${fill.repeat(3)}</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>`);
write('ppt/notesMasters/notesMaster1.xml', `<p:notesMaster ${header}><p:cSld><p:spTree>${group}</p:spTree></p:cSld>${colorMap}<p:notesStyle/></p:notesMaster>`);
rels('ppt/notesMasters/_rels/notesMaster1.xml.rels', [{id:'rId1',type:'theme',target:'../theme/theme1.xml'}]);
for (let i=1; i<=slides.length; i++) {
  const {info,page,background,texts}=slides[i-1];
  assert(texts.length && texts.some(t=>t.runs.some(r=>r.text.includes(`Page ${page}`))),`Missing editable page label on slide ${i}`);
  const png=`background-${i}.png`;
  media(`ppt/media/${png}`,path.join(build,background));
  const pics=picture(2,`${info.title} graphic elements`,'rId2',0,0,1280,720)+texts.map((t,index)=>textBox(index+3,t)).join('');
  const links=[{id:'rId1',type:'slideLayout',target:'../slideLayouts/slideLayout1.xml'},{id:'rId2',type:'image',target:`../media/${png}`},{id:'rId4',type:'notesSlide',target:`../notesSlides/notesSlide${i}.xml`}];
  write(`ppt/slides/slide${i}.xml`,`<p:sld ${header}><p:cSld name="${xml(info.title)}"><p:spTree>${group}${pics}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>`);
  rels(`ppt/slides/_rels/slide${i}.xml.rels`,links);
  const paragraphs=(info.title+'\n\n'+info.notes).split('\n').map(t=>`<a:p><a:r><a:rPr lang="en-US" sz="1400"><a:latin typeface="Arial"/></a:rPr><a:t xml:space="preserve">${xml(t)}</a:t></a:r><a:endParaRPr lang="en-US"/></a:p>`).join('');
  write(`ppt/notesSlides/notesSlide${i}.xml`,`<p:notes ${header}><p:cSld><p:spTree>${group}<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>${paragraphs}</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>`);
  rels(`ppt/notesSlides/_rels/notesSlide${i}.xml.rels`,[{id:'rId1',type:'slide',target:`../slides/slide${i}.xml`},{id:'rId2',type:'notesMaster',target:'../notesMasters/notesMaster1.xml'}]);
}
for (const name of files) {
  if (!name.endsWith('.rels')) continue;
  const dir=path.posix.dirname(name).replace(/(?:^|\/)\_rels$/,'');
  const text=fs.readFileSync(path.join(pkg,name),'utf8');
  for (const match of text.matchAll(/Target="([^"]+)"/g)) assert(files.has(path.posix.normalize(path.posix.join(dir,match[1]))),`Missing relationship from ${name}: ${match[1]}`);
}
assert(![...files].some(name => name.endsWith('.gif')));
const candidate=path.join(build,`${path.basename(pkg)}.pptx`);
execFileSync('/usr/bin/zip',['-q','-r',candidate,'.'],{cwd:pkg});
execFileSync('/usr/bin/unzip',['-tq',candidate],{stdio:'inherit'});
fs.copyFileSync(candidate,path.join(out,'patent-v2.pptx'));
console.log(`Exported ${slides.length} static slides with ${slides.reduce((n,s)=>n+s.texts.length,0)} editable text boxes and original page labels.`);

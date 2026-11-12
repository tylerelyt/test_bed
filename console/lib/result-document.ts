// Shared with isolated original HTML results, including nested graph documents.
const theme = `
:root{color-scheme:light;--result-ink:#292d28;--result-muted:#767d73;--result-line:#e4e7df;--result-surface:#f7f8f4;--result-accent:#b65b3c}
*{box-sizing:border-box}
html,body{margin:0!important;padding:0!important;background:#fff!important;color:var(--result-ink)!important;font-family:Arial,"PingFang SC","Microsoft YaHei",sans-serif!important;font-size:13px!important;line-height:1.75!important}
#native-html-content{display:flow-root;padding:0;min-width:0;overflow-wrap:anywhere}
h1,h2,h3,h4,h5,h6{font-family:inherit!important;color:var(--result-ink)!important;font-weight:600!important;letter-spacing:0!important;line-height:1.5!important;margin:0 0 14px!important}
h1,h2{font-size:18px!important}h3,h4{font-size:15px!important}h5,h6{font-size:12px!important}
p{margin:8px 0!important}ul,ol{padding-left:20px!important;margin:10px 0!important}li{margin:4px 0}
a{color:var(--result-accent)!important;text-underline-offset:3px}
[style*="background"]{background-color:var(--result-surface)!important}
[style*="border"]{border-color:var(--result-line)!important;border-radius:8px!important}
[style*="color"]:not([data-result-tone]){color:var(--result-ink)!important}
[data-result-tone="success"]{color:#45694e!important}[data-result-tone="warning"]{color:#936725!important}[data-result-tone="error"]{color:#a8463e!important}
[data-result-surface="success"]{background-color:#eff5ef!important;border-color:#d8e6d8!important}
[data-result-surface="warning"]{background-color:#fbf5e9!important;border-color:#eadfc5!important}
[data-result-surface="error"]{background-color:#fbefed!important;border-color:#efd9d4!important}
pre,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace!important;font-size:12px!important}
pre{white-space:pre-wrap!important;overflow-wrap:anywhere;padding:16px!important;background:var(--result-surface)!important;border:1px solid var(--result-line)!important;border-radius:8px!important;max-width:100%}
table{width:100%!important;border-collapse:collapse!important;font-size:12px!important}th,td{padding:11px 13px!important;text-align:left;border-bottom:1px solid var(--result-line)!important}th{background:var(--result-surface)!important;color:var(--result-muted)!important;font-weight:500!important}
img,video,svg{max-width:100%}iframe{display:block;max-width:100%;border:0!important;border-radius:8px!important}
.wrap{border:1px solid var(--result-line)!important;border-radius:8px!important}.toolbar{background:var(--result-surface)!important;border-bottom:1px solid var(--result-line)!important;padding:12px 16px!important}
button{font:inherit;border:1px solid var(--result-line);border-radius:6px;background:#fff;padding:8px 12px;color:var(--result-ink)}
@media(max-width:650px){[style*="padding"]{max-width:100%}table{display:block;overflow:auto}th,td{padding:8px!important}}
`;

export function resultDocument(html: string): string {
  const localHtml = html
    .replaceAll('https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js', '/vendor/vis-network/vis-network.min.js')
    .replaceAll('https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css', '/vendor/vis-network/vis-network.min.css');
  const style = `<style data-workbench-theme>${theme}</style>`;
  // Classify existing semantic colors before overriding the old inline palette.
  // Nested srcdoc is themed before loading, so graph scripts execute just once.
  const script = `
const theme=${JSON.stringify(theme)};
function decorate(doc){
 for(const el of doc.querySelectorAll('[style]')){
  const color=(el.style.color+' '+el.style.backgroundColor+' '+el.style.borderColor).toLowerCase();
  let tone='';
  if(/green|76, 175, 80|40, 167, 69|21, 87, 36|212, 237, 218/.test(color))tone='success';
  else if(/red|220, 53, 69|114, 28, 36|248, 215, 218/.test(color))tone='error';
  else if(/orange|255, 152, 0|255, 193, 7/.test(color))tone='warning';
  if(tone){el.dataset.resultTone=tone;if(el.style.backgroundColor||el.style.borderColor)el.dataset.resultSurface=tone;}
 }
 for(const el of doc.querySelectorAll('h1,h2,h3,h4,h5,h6')){
  const node=el.firstChild;
  if(node&&node.nodeType===3)node.textContent=node.textContent.replace(/^(?:[\\p{Extended_Pictographic}\\uFE0F\\u200D]|\\s)+/u,'');
 }
 for(const frame of doc.querySelectorAll('iframe[srcdoc]')){
  const nested=new DOMParser().parseFromString(frame.getAttribute('srcdoc'),'text/html');
  decorate(nested);const style=nested.createElement('style');style.textContent=theme;nested.head.appendChild(style);
  frame.setAttribute('srcdoc','<!doctype html>'+nested.documentElement.outerHTML);
 }
}
decorate(document);
const root=document.getElementById('native-html-content');
const measure=()=>parent.postMessage({height:Math.ceil(root.scrollHeight)},'*');
new ResizeObserver(measure).observe(root);
addEventListener('message',e=>{if(e.data?.type==='native-measure')measure()});
addEventListener('load',measure);measure();`;
  return `${style}<div id="native-html-content">${localHtml}</div><script>${script}</script>`;
}

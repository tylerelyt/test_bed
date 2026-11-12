// Presentation only: original values and callback payloads are never changed.
export function displayLabel(value: string): string {
  return value
    .replace(/^([0-9])\uFE0F?\u20E3\s*/u, '$1 · ')
    .replace(/^(?:[\p{Extended_Pictographic}\p{Regional_Indicator}\uFE0F\u200D\u20E3]|\s)+/u, '')
    .trim();
}

export function displayMarkdown(value: string, moduleIntro = false): string {
  let fenced = false;
  let firstHeading = true;
  return value.split('\n').map(line => {
    if (/^\s*(```|~~~)/.test(line)) { fenced = !fenced; return line; }
    if (fenced || !/^#{1,6}\s/.test(line)) return line;
    if (moduleIntro && firstHeading) { firstHeading = false; return ''; }
    firstHeading = false;
    return line.replace(/^(#{1,6})\s+(.+)$/, (_, marks, title) => `${marks} ${displayLabel(title)}`);
  }).join('\n').trim();
}

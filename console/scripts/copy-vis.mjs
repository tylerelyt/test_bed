import {mkdir,copyFile,cp} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const source=path.join(root,'node_modules/vis-network');
const target=path.join(root,'public/vendor/vis-network');
await mkdir(target,{recursive:true});
for(const [from,to]of [
 ['standalone/umd/vis-network.min.js','vis-network.min.js'],
 ['styles/vis-network.min.css','vis-network.min.css'],
 ['LICENSE-MIT','LICENSE-MIT'],['LICENSE-APACHE-2.0','LICENSE-APACHE-2.0'],
])await copyFile(path.join(source,from),path.join(target,to));
await cp(path.join(source,'dist/img'),path.join(target,'img'),{recursive:true});

import type { NextConfig } from 'next';
const config: NextConfig = {
 poweredByHeader: false,
 async headers(){return [{source:'/vendor/vis-network/:path*',headers:[{key:'Access-Control-Allow-Origin',value:'*'}]}]},
};
export default config;

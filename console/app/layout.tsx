import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {title:'Test Bed · FDE 工程工作台', description:'配置企业上下文来源，为岗位 Agent 供应事实、方法、工具与历史'};
export default function Layout({children}:{children:React.ReactNode}) {return <html lang="zh-CN"><body>{children}</body></html>}

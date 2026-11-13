import {readFile} from 'node:fs/promises';

// One explicit tool, with no caller-selected path, URL or command.
export default function enterpriseContext(pi) {
  pi.registerTool({
    name: 'read_enterprise_context',
    label: '读取岗位上下文',
    description: '读取后台为本次岗位 Agent 固定的企业资料与来源；回答业务问题前先读取。',
    parameters: {type: 'object', properties: {}, additionalProperties: false},
    async execute() {
      const body = await readFile(process.env.WORKBENCH_CONTEXT_FILE, 'utf8');
      return {content: [{type: 'text', text: body}], details: {source: 'published-context-package'}};
    },
  });
}

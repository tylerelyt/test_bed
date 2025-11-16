---
layout: default
title: Troubleshooting
parent: GUI Automation Agent
nav_order: 4
---

# Troubleshooting

Common issues and solutions for GUI automation.

## VM Issues

### VM Won't Start

```bash
# Check Docker
docker ps -a | grep osworld

# View logs
docker logs osworld-vm

# Restart container
docker restart osworld-vm
```

---

## Permission Issues (macOS)

### Accessibility Permission

1. Open System Settings
2. Privacy & Security → Accessibility
3. Add Terminal/Python
4. Restart application

### Screen Recording Permission

1. Open System Settings
2. Privacy & Security → Screen Recording
3. Add Terminal/Python
4. Restart application

---

## Model Issues

### API Call Failed

**Check**:
- ✅ API Key correct
- ✅ Network connection OK
- ✅ API quota sufficient

**Qwen-VL Test**:
```bash
curl -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -d '{"model": "qwen3-vl-plus", "messages": [{"role": "user", "content": "test"}]}'
```

---

## Performance Issues

### Slow Response

**Solutions**:
- Use faster model (qwen3-vl-flash)
- Reduce max_steps
- Lower screenshot resolution

### Actions Not Executing

**Check**:
- Verify coordinates are within screen bounds
- Ensure target window is visible
- Check accessibility permissions

---
layout: default
title: Environment Setup
parent: GUI Automation Agent
nav_order: 1
---

# Environment Setup

Configure desktop environment for GUI automation.

## VM Mode (Recommended)

### Start Docker Container

```bash
docker run -d \
  --name osworld-vm \
  -p 55000:5000 \
  -p 5901:5900 \
  xlangai/osworld:latest
```

### Web UI Operations

1. Navigate to "🤖 GUI-Agent" tab
2. Click "🚀 Start VM"
3. Wait for "✅ Running" status

---

## Local Mode

### macOS Permissions

```
System Settings → Privacy & Security → Accessibility
- Add Terminal/Python

System Settings → Privacy & Security → Screen Recording
- Add Terminal/Python
```

### Web UI Operations

1. Select "Local System (Local)"
2. Configure model and API Key
3. Start task execution

---

## Verification

```bash
# Test screenshot
python -c "import pyautogui; pyautogui.screenshot()"

# Check Ollama (if using local models)
curl http://localhost:11434/api/tags
```

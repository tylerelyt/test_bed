"""Project-owned immutable uploads; business services never accept download paths."""
import hashlib
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree

from ..workbench_access import private_write


def save_json(path, value):
    private_write(path, json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path, suffix):
    if suffix in ('.txt', '.md', '.csv', '.json'):
        result = path.read_text(encoding='utf-8-sig')
    elif suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if len(reader.pages) > 500:
            raise ValueError('单次导入最多 500 页')
        result = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
    elif suffix == '.docx':
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > 20_000_000:
                raise ValueError('文档解压后过大')
            root = ElementTree.fromstring(archive.read(info))
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        result = '\n'.join(''.join(p.itertext()) for p in root.iter(ns+'p'))
    else:
        raise ValueError('支持 UTF-8 文本、Markdown、CSV、JSON、PDF、DOCX')
    if not result.strip():
        raise ValueError('未提取到文本；扫描件需要另行 OCR')
    if len(result) > 200000:
        raise ValueError('提取内容超过 20 万字符，请拆分资料')
    return result

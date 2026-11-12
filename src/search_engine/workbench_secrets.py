"""Redact known credentials from browser-facing callback values and layouts."""
import os
import re

SECRET_LABEL = re.compile(r'password|secret|credential|api[_ -]?key|access[_ -]?token|密码|密钥|令牌', re.I)
HIDDEN = '【已隐藏】'


def private_field(props):
    return props.get('type') == 'password' or bool(SECRET_LABEL.search(str(props.get('label', ''))))


def public_value(value, extra=()):
    secrets = {item for item in extra if isinstance(item, str) and item}
    secrets.update(content for name, content in os.environ.items() if content and
                   (SECRET_LABEL.search(name) or re.search(r'(?:^|_)TOKEN$', name)))
    def clean(item):
        if isinstance(item, dict):
            return {key: HIDDEN if SECRET_LABEL.search(str(key)) and not isinstance(content, bool) else clean(content)
                    for key, content in item.items()}
        if isinstance(item, list):
            return [clean(content) for content in item]
        if isinstance(item, str):
            for secret in sorted(secrets, key=len, reverse=True):
                item = item.replace(secret, HIDDEN)
        return item
    return clean(value)

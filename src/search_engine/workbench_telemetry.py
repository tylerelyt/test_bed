"""Collect provider-reported usage in the current callback, without changing its output."""
import contextvars
import math
import os
import json
from contextlib import contextmanager

_current = contextvars.ContextVar('workbench_usage', default=None)


@contextmanager
def observe_usage():
    calls = []
    token = _current.set(calls)
    try:
        yield calls
    finally:
        _current.reset(token)


def record_usage(response, provider='dashscope'):
    calls = _current.get()
    if calls is None:
        return
    usage = getattr(response, 'usage', None)
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return
    model = str(getattr(response, 'model', 'unknown'))
    incoming = usage.get('prompt_tokens', usage.get('input_tokens'))
    outgoing = usage.get('completion_tokens', usage.get('output_tokens'))
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (incoming, outgoing)):
        return
    cost = None
    price = None
    try:
        price = json.loads(os.environ.get('WORKBENCH_MODEL_PRICES', '{}')).get(model)
        if price and isinstance(price.get('source'), str) and price['source'].strip() and isinstance(price.get('currency'), str) and price['currency'].strip():
            rates = [price[k] for k in ('input_per_million', 'output_per_million')]
            if all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) and n >= 0 for n in rates):
                cost = (incoming * rates[0] + outgoing * rates[1]) / 1_000_000
    except (ValueError, TypeError, KeyError, AttributeError):
        price = None
    calls.append({'provider': provider, 'model': model, 'input_tokens': incoming,
                  'output_tokens': outgoing, 'cost': cost,
                  'currency': price.get('currency') if cost is not None else None,
                  'price_source': price.get('source') if cost is not None else None})

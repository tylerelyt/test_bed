#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 OpenKG 电影类 JSON 数据转换为 Janus 导入可用的 TSV 三元组。

输出格式固定为：
subject<TAB>predicate<TAB>object

默认目标是 OpenKG 的 douban-movie-kg 资源（dbmovies.json）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_DOUBAN_URL = (
    "http://data.openkg.cn/dataset/3de0675f-660c-4b00-800c-f07b479c2c19/"
    "resource/be21c7c4-78a4-416f-b8aa-8d4bf6d2ae0d/download/dbmovies.json"
)
DEFAULT_OUTPUT = "data/openkg_triples.tsv"
DEFAULT_RAW_CACHE = "data/openkg_douban_movie.json"

# 仅保留电影域常见关系，避免把噪声字段直接灌入图谱
PREDICATE_ALLOWLIST = {
    "导演",
    "编剧",
    "主演",
    "演员",
    "类型",
    "制片国家",
    "国家",
    "地区",
    "语言",
    "上映年份",
    "上映日期",
    "片长",
    "豆瓣评分",
    "IMDb",
    "别名",
}

MOVIE_TITLE_KEYS = (
    "电影名",
    "片名",
    "title",
    "name",
    "movie_name",
    "movieTitle",
)

PREDICATE_KEY_MAP = {
    "director": "导演",
    "directors": "导演",
    "导演": "导演",
    "writer": "编剧",
    "writers": "编剧",
    "screenwriter": "编剧",
    "编剧": "编剧",
    "actor": "演员",
    "actors": "演员",
    "cast": "主演",
    "starring": "主演",
    "主演": "主演",
    "genre": "类型",
    "genres": "类型",
    "类型": "类型",
    "category": "类型",
    "country": "国家",
    "countries": "国家",
    "地区": "地区",
    "district": "地区",
    "language": "语言",
    "languages": "语言",
    "上映年份": "上映年份",
    "year": "上映年份",
    "showtime": "上映年份",
    "上映日期": "上映日期",
    "release_date": "上映日期",
    "runtime": "片长",
    "片长": "片长",
    "length": "片长",
    "douban_rating": "豆瓣评分",
    "rating": "豆瓣评分",
    "rate": "豆瓣评分",
    "评分": "豆瓣评分",
    "imdb": "IMDb",
    "alias": "别名",
    "aka": "别名",
    "othername": "别名",
    "别名": "别名",
    "composer": "编剧",
}


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _iter_values(v: Any) -> Iterable[str]:
    if v is None:
        return []
    if isinstance(v, list):
        for item in v:
            t = _normalize_text(item)
            if t:
                yield t
        return
    text = _normalize_text(v)
    if not text:
        return
    if any(sep in text for sep in [" / ", "/", "、", "|", ";", "；", ",", "，"]):
        parts = re.split(r"\s*(?:/|、|\||;|；|,|，)\s*", text)
        for p in parts:
            q = _normalize_text(p)
            if q:
                yield q
    else:
        yield text


def _download_with_retries(url: str, dest: str, retries: int = 6, timeout: int = 90) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last_err: Exception | None = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            with open(dest, "wb") as f:
                f.write(body)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if i < retries:
                time.sleep(min(2 * i, 10))
    raise RuntimeError(f"下载失败: {url}; last_error={last_err}")


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "items", "movies", "results"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # 兼容单对象
        return [data]
    return []


def _pick_movie_title(rec: Dict[str, Any]) -> str:
    for k in MOVIE_TITLE_KEYS:
        if k in rec:
            title = _normalize_text(rec.get(k))
            if title:
                return title
    # 兜底：若无显式标题，尝试主键字段
    for k in ("id", "_id"):
        if k in rec:
            title = _normalize_text(rec.get(k))
            if title:
                return title
    return ""


def _to_predicate(key: str) -> str:
    k = _normalize_text(key)
    return PREDICATE_KEY_MAP.get(k, k)


def _record_to_triples(rec: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    subject = _pick_movie_title(rec)
    if not subject:
        return []
    out: List[Tuple[str, str, str]] = []
    for raw_key, raw_value in rec.items():
        if raw_key in MOVIE_TITLE_KEYS or raw_key in ("id", "_id", "url", "cover"):
            continue
        predicate = _to_predicate(raw_key)
        if predicate not in PREDICATE_ALLOWLIST:
            continue
        for obj in _iter_values(raw_value):
            tri = (_normalize_text(subject), _normalize_text(predicate), _normalize_text(obj))
            if tri[0] and tri[1] and tri[2]:
                out.append(tri)
    return out


def _match_genre_filter(rec: Dict[str, Any], genre_keyword: str) -> bool:
    kw = _normalize_text(genre_keyword)
    if not kw:
        return True
    candidates: List[str] = []
    for gk in ("category", "类型", "genre", "genres"):
        if gk in rec and rec[gk] is not None:
            v = rec[gk]
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                s = _normalize_text(x)
                if s:
                    candidates.extend(re.split(r"\s*(?:/|、|\||;|；|,|，)\s*", s))
    return any(kw in _normalize_text(x) for x in candidates)


def convert_records_to_triples(
    records: Sequence[Dict[str, Any]],
    max_records: int = 8000,
    genre_keyword: str = "",
) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for rec in records[:max_records]:
        if not _match_genre_filter(rec, genre_keyword):
            continue
        for tri in _record_to_triples(rec):
            if tri in seen:
                continue
            seen.add(tri)
            triples.append(tri)
    return triples


def build_scifi_closure_triples(
    records: Sequence[Dict[str, Any]],
    max_records: int = 100000,
    seed_genre: str = "科幻",
) -> List[Tuple[str, str, str]]:
    """
    以 seed_genre 电影为种子，按“实体-关系-实体”无向连通做外延闭包，
    返回闭包内全部三元组。
    """
    all_triples: List[Tuple[str, str, str]] = []
    seen_triples: Set[Tuple[str, str, str]] = set()
    all_movies: Set[str] = set()
    for rec in records[:max_records]:
        title = _pick_movie_title(rec)
        if title:
            all_movies.add(title)
        for tri in _record_to_triples(rec):
            if tri in seen_triples:
                continue
            seen_triples.add(tri)
            all_triples.append(tri)

    seeds: Set[str] = set()
    for s, p, o in all_triples:
        if p == "类型" and o == seed_genre:
            seeds.add(s)

    if not seeds:
        return []

    adj: Dict[str, Set[str]] = defaultdict(set)
    for s, _p, o in all_triples:
        adj[s].add(o)
        adj[o].add(s)

    reached: Set[str] = set(seeds)
    stack: List[str] = list(seeds)
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            # 关键约束：闭包不能扩展到非科幻电影节点
            if nxt in all_movies and nxt not in seeds:
                continue
            if nxt in reached:
                continue
            reached.add(nxt)
            stack.append(nxt)
    # 再做一层硬过滤：任一端点若是电影，必须属于科幻种子电影
    out: List[Tuple[str, str, str]] = []
    for t in all_triples:
        s, _p, o = t
        if s not in reached or o not in reached:
            continue
        if s in all_movies and s not in seeds:
            continue
        if o in all_movies and o not in seeds:
            continue
        out.append(t)
    return out


def write_tsv(triples: Sequence[Tuple[str, str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenKG 电影 JSON -> openkg_triples.tsv")
    parser.add_argument("--source-url", default=DEFAULT_DOUBAN_URL, help="OpenKG 数据资源 URL")
    parser.add_argument("--cache-json", default=DEFAULT_RAW_CACHE, help="下载后缓存 JSON 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 TSV 路径")
    parser.add_argument("--max-records", type=int, default=8000, help="最多处理记录数")
    parser.add_argument("--reuse-cache", action="store_true", help="若缓存文件存在则跳过下载")
    parser.add_argument(
        "--genre-keyword",
        default="",
        help="按电影类型关键词过滤（例如: 科幻）。留空表示不过滤。",
    )
    parser.add_argument(
        "--scifi-closure",
        action="store_true",
        help="以科幻电影为种子，导出外延关系闭包（连通闭包）。",
    )
    parser.add_argument(
        "--seed-genre",
        default="科幻",
        help="闭包模式下的种子类型，默认“科幻”。",
    )
    args = parser.parse_args()

    if not (args.reuse_cache and os.path.exists(args.cache_json)):
        print(f"Downloading: {args.source_url}")
        _download_with_retries(args.source_url, args.cache_json)
        print(f"Cached JSON: {args.cache_json}")
    else:
        print(f"Reuse cache JSON: {args.cache_json}")

    records = _load_records(args.cache_json)
    print(f"Loaded records: {len(records)}")
    if args.scifi_closure:
        print(f"Closure mode enabled, seed genre: {args.seed_genre}")
        triples = build_scifi_closure_triples(
            records,
            max_records=args.max_records,
            seed_genre=args.seed_genre,
        )
    else:
        if args.genre_keyword:
            print(f"Genre filter keyword: {args.genre_keyword}")
        triples = convert_records_to_triples(
            records,
            max_records=args.max_records,
            genre_keyword=args.genre_keyword,
        )
    print(f"Converted triples: {len(triples)}")
    write_tsv(triples, args.output)
    print(f"Wrote TSV: {args.output}")


if __name__ == "__main__":
    main()

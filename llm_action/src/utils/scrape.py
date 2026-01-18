import os
import json
from typing import List, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md

from llm_action.src.models import DocTreeNode, Documentation

# parsing utilities

def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def is_heading(t: Tag) -> bool:
    return isinstance(t, Tag) and t.name in {"h1", "h2", "h3", "h4"}

def level(t: Tag) -> int:
    return int(t.name[1])

def main_container(soup: BeautifulSoup) -> Tag:
    c = soup.select_one("div.document") or soup.select_one("div[role='main']") or soup.body
    if c is None:
        raise RuntimeError("No main container found")
    return c

def linear_tags(root: Tag) -> List[Tag]:
    return [t for t in root.descendants if isinstance(t, Tag)]

def blocks(tags: List[Tag]) -> List[Dict[str, Any]]:
    hp = [i for i, t in enumerate(tags) if is_heading(t)]
    if not hp:
        raise RuntimeError("No headings found")

    out: List[Dict[str, Any]] = []
    for k, i in enumerate(hp):
        h = tags[i]
        j = hp[k + 1] if k + 1 < len(hp) else len(tags)

        content_tags: List[Tag] = []
        for x in range(i + 1, j):
            if is_heading(tags[x]):
                break
            content_tags.append(tags[x])

        name = h.get_text(" ", strip=True)
        html_block = str(h) + "\n" + "\n".join(str(t) for t in content_tags)
        content = md(html_block, heading_style="ATX").strip()
        out.append({"name": name, "level": level(h), "content": content})

    return out

def build_tree(flat_blocks: List[Dict[str, Any]]) -> DocTreeNode:
    root: Dict[str, Any] = {"name": "ROOT", "level": 0, "content": "", "children": []}
    stack: List[Dict[str, Any]] = [root]

    for b in flat_blocks:
        node: Dict[str, Any] = {"name": b["name"], "level": b["level"], "content": b["content"], "children": []}
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        parent = stack[-1] if stack else root
        parent["children"].append(node)
        stack.append(node)

    return DocTreeNode(**root)

# cleaning utilities

def join_md(parts: List[str]) -> str:
    parts = [p.strip() for p in parts if p and p.strip()]
    return "\n\n---\n\n".join(parts).strip()

def collect_md_tree(node: DocTreeNode) -> str:
    parts: List[str] = []
    if node.content.strip():
        parts.append(node.content.strip())
    for ch in node.children:
        md_text = collect_md_tree(ch)
        if md_text:
            parts.append(md_text)
    return join_md(parts)

def collect_md_doc(doc: Documentation) -> str:
    parts: List[str] = []

    for category in doc.transformation_categories:
        category_parts: List[str] = []

        for t in category.transformations:
            if t.content.strip():
                category_parts.append(t.content.strip())

        if category_parts:
            parts.append("\n\n---\n\n".join(category_parts))

    return "\n\n---\n\n".join(parts).strip()

def prune_at_level(root: DocTreeNode, target_level: int, depth: int) -> DocTreeNode:
    if depth < 0:
        raise ValueError("depth must be >= 0")

    def prune_relative(node: DocTreeNode, d: int) -> DocTreeNode:
        out = DocTreeNode(name=node.name, level=node.level, content=node.content or "", children=[])
        if d == 0:
            merged: List[str] = [out.content]
            for ch in node.children:
                merged.append(collect_md_tree(ch))
            out.content = join_md(merged)
            out.children = []
            return out
        out.children = [prune_relative(ch, d - 1) for ch in node.children]
        return out

    def walk(node: DocTreeNode) -> DocTreeNode:
        if node.level == target_level:
            return prune_relative(node, depth)
        return DocTreeNode(
            name=node.name,
            level=node.level,
            content=node.content or "",
            children=[walk(ch) for ch in node.children],
        )

    return walk(root)

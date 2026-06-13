"""Table structure metric: TEDS (Tree-Edit-Distance-based Similarity).

Standard TEDS (Zhong et al., PubTabNet): parse both tables to trees, compute
tree-edit distance with APTED where same-tag/span nodes cost 0, td/th leaves cost
a normalized Levenshtein on their text, and TEDS = 1 - dist / max(nodes). HTML is
parsed with the stdlib html.parser (no lxml). 1.0 = identical structure+content.
"""

from __future__ import annotations

from html.parser import HTMLParser

from apted import APTED, Config


class _Node:
    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag, colspan=1, rowspan=1):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = ""
        self.children: list["_Node"] = []


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = _Node("__root__")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        node = _Node(tag, int(a.get("colspan", 1) or 1), int(a.get("rowspan", 1) or 1))
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag):
        if len(self.stack) > 1:
            self.stack.pop()

    def handle_data(self, data):
        if self.stack[-1].tag in ("td", "th"):
            self.stack[-1].content += data


def _parse(html: str) -> _Node:
    p = _TableParser()
    p.feed(html or "")
    # The <table> node if present, else the synthetic root.
    for c in p.root.children:
        if c.tag == "table":
            return c
    return p.root


def _count(node: _Node) -> int:
    return 1 + sum(_count(c) for c in node.children)


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class _TEDSConfig(Config):
    def rename(self, n1, n2):
        if n1.tag != n2.tag or n1.colspan != n2.colspan or n1.rowspan != n2.rowspan:
            return 1.0
        if n1.tag in ("td", "th") and (n1.content or n2.content):
            m = max(len(n1.content), len(n2.content))
            return _lev(n1.content.strip(), n2.content.strip()) / m if m else 0.0
        return 0.0

    def children(self, node):
        return node.children


def teds(html_pred: str, html_ref: str) -> float:
    t1, t2 = _parse(html_pred), _parse(html_ref)
    n1, n2 = _count(t1), _count(t2)
    if n1 <= 1 and n2 <= 1:
        return 1.0 if (n1 == n2) else 0.0
    dist = APTED(t1, t2, _TEDSConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / max(n1, n2))

"""Generate the system design diagram as an editable Excalidraw file and an SVG (same layout).

    .venv/bin/python docs/make_diagram.py
Open docs/system_design.excalidraw at excalidraw.com (File > Open) to edit.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).parent
W, H = 1700, 1000

# (id, x, y, w, h, title, subtitle, colour)
C = {"data": "#e7f5ff", "llm": "#fff4e6", "ml": "#ebfbee", "check": "#fff0f6", "ui": "#f3f0ff", "store": "#f8f9fa"}
BOXES = [
    # lane 1: offline data
    ("wiki", 40, 90, 230, 80, "Fandom wikis", "MediaWiki API, ~80 pages, CC BY-SA", "data"),
    ("corpus", 320, 90, 230, 80, "Canon corpus", "text only, URL + revision kept", "data"),
    ("index", 600, 90, 260, 80, "Hybrid index", "chunks: BM25 + Gemini embeddings", "data"),
    ("cards", 910, 90, 260, 80, "8 CS concept cards", "roles, relations, misconceptions, quiz", "data"),
    ("db", 1220, 90, 260, 80, "SQLite", "LLM cache + run logs (replayable demo)", "store"),
    # lane 2: request path
    ("user", 40, 330, 200, 80, "User", "anime + CS topic", "ui"),
    ("brain", 280, 330, 210, 80, "1. Brainstorm", "5 story elements (LLM)", "llm"),
    ("ret", 530, 330, 210, 80, "2. Retrieve canon", "hybrid search per element", "data"),
    ("cand", 780, 330, 230, 80, "3. Candidates", "5 structured mappings + citations", "llm"),
    ("feat", 1050, 330, 200, 80, "4. Features", "coverage, support, ...", "ml"),
    ("rank", 1290, 330, 230, 80, "5. Quality classifier", "logistic regression / RF", "ml"),
    ("write", 1290, 520, 230, 80, "6. Lesson writer", "LLM, cites every anime fact", "llm"),
    ("gate", 1000, 520, 250, 80, "7. Faithfulness gate", "claims checked vs canon + card", "check"),
    ("app", 660, 520, 300, 80, "8. Streamlit app", "lesson, mapping, breaks, quiz, sources", "ui"),
    # lane 3: learning loop + evidence
    ("label", 280, 790, 230, 80, "Labelling tool", "~200 candidates rated blind", "ml"),
    ("labels", 560, 790, 200, 80, "labels.csv", "good / ok / bad + error flags", "store"),
    ("train", 840, 790, 250, 80, "Train + grouped CV", "vs embedding + self-rating baselines", "ml"),
    ("evid", 1150, 790, 300, 80, "Evidence page", "ablation A-D, ROC, coefficients", "ui"),
]
ARROWS = [("wiki", "corpus"), ("corpus", "index"), ("user", "brain"), ("brain", "ret"), ("ret", "cand"),
          ("cand", "feat"), ("feat", "rank"), ("rank", "write"), ("write", "gate"), ("gate", "app"),
          ("index", "ret"), ("cards", "cand"), ("label", "labels"), ("labels", "train"), ("train", "evid")]
# elbow-routed arrows: explicit points from start to end
ROUTED = [
    [(1100, 170), (1100, 250), (1270, 250), (1270, 470), (1125, 470), (1125, 520)],   # cards -> gate
    [(895, 410), (895, 460), (395, 460), (395, 790)],                                 # cand -> label
    [(965, 790), (965, 700), (1610, 700), (1610, 370), (1520, 370)],                  # train -> rank
]
LANES = [(60, "Offline: build the knowledge"), (300, "Request: anime + topic to a checked lesson"),
         (760, "Learning loop")]
B = {b[0]: b for b in BOXES}


def anchor(a, b):
    ax, ay, aw, ah = B[a][1:5]
    bx, by, bw, bh = B[b][1:5]
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    if abs(acx - bcx) * ah > abs(acy - bcy) * aw:  # mostly horizontal
        return (ax + aw if bcx > acx else ax, acy), (bx if bcx > acx else bx + bw, bcy)
    return (acx, ay + ah if bcy > acy else ay), (bcx, by if bcy > acy else by + bh)


def svg():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="ar" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
         '<path d="M0,0 L10,4 L0,8 z" fill="#495057"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         '<text x="40" y="44" font-size="28" font-weight="700" fill="#212529">Canonical: system design</text>']
    for y, t in LANES:
        s.append(f'<text x="40" y="{y + 18}" font-size="15" font-weight="700" fill="#868e96" '
                 f'letter-spacing="1">{t.upper()}</text>')
    for a, b in ARROWS:
        (x1, y1), (x2, y2) = anchor(a, b)
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#495057" stroke-width="2" '
                 f'marker-end="url(#ar)"/>')
    for pts in ROUTED:
        s.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in pts) + '" fill="none" stroke="#495057" '
                 'stroke-width="2" marker-end="url(#ar)"/>')
    for i, x, y, w, h, t, sub, col in BOXES:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{C[col]}" stroke="#343a40" '
                 f'stroke-width="2"/>')
        s.append(f'<text x="{x + w / 2}" y="{y + 34}" font-size="18" font-weight="700" text-anchor="middle" '
                 f'fill="#212529">{t}</text>')
        s.append(f'<text x="{x + w / 2}" y="{y + 58}" font-size="13" text-anchor="middle" fill="#495057">{sub}</text>')
    lx = 40
    for name, col in [("data / retrieval", "data"), ("LLM step", "llm"), ("classical ML (course)", "ml"),
                      ("verification", "check"), ("interface", "ui"), ("storage", "store")]:
        s.append(f'<rect x="{lx}" y="{H - 50}" width="22" height="22" rx="4" fill="{C[col]}" stroke="#343a40"/>')
        s.append(f'<text x="{lx + 30}" y="{H - 33}" font-size="14" fill="#495057">{name}</text>')
        lx += 210
    s.append("</svg>")
    (OUT / "system_design.svg").write_text("\n".join(s))


def excalidraw():
    rnd = random.Random(1)
    els = []

    def base(**kw):
        d = {"angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
             "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
             "frameId": None, "roundness": None, "seed": rnd.randint(1, 2**31), "version": 1,
             "versionNonce": rnd.randint(1, 2**31), "isDeleted": False, "boundElements": [], "updated": 1,
             "link": None, "locked": False}
        d.update(kw)
        return d

    def text(x, y, w, t, size, bold=False, align="center", container=None, colour="#1e1e1e"):
        return base(type="text", id=f"t{len(els)}", x=x, y=y, width=w, height=size * 1.3, text=t,
                    originalText=t, fontSize=size, fontFamily=2 if not bold else 2, textAlign=align,
                    verticalAlign="top", containerId=container, lineHeight=1.25, autoResize=True,
                    strokeColor=colour, baseline=int(size))

    els.append(text(40, 20, 600, "Canonical: system design", 28, True, "left"))
    for y, t in LANES:
        els.append(text(40, y, 700, t.upper(), 16, True, "left", colour="#868e96"))
    for i, x, y, w, h, t, sub, col in BOXES:
        els.append(base(type="rectangle", id=i, x=x, y=y, width=w, height=h, backgroundColor=C[col],
                        roundness={"type": 3}))
        els.append(text(x, y + 14, w, t, 20, True))
        els.append(text(x, y + 46, w, sub, 14, colour="#495057"))
    for a, b in ARROWS:
        (x1, y1), (x2, y2) = anchor(a, b)
        els.append(base(type="arrow", id=f"a-{a}-{b}", x=x1, y=y1, width=abs(x2 - x1), height=abs(y2 - y1),
                        points=[[0, 0], [x2 - x1, y2 - y1]], startBinding={"elementId": a, "focus": 0, "gap": 4},
                        endBinding={"elementId": b, "focus": 0, "gap": 4}, startArrowhead=None,
                        endArrowhead="arrow", roundness={"type": 2}, lastCommittedPoint=None))
    for k, pts in enumerate(ROUTED):
        x0, y0 = pts[0]
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        els.append(base(type="arrow", id=f"r{k}", x=x0, y=y0, width=max(xs) - min(xs), height=max(ys) - min(ys),
                        points=[[x - x0, y - y0] for x, y in pts], startBinding=None, endBinding=None,
                        startArrowhead=None, endArrowhead="arrow", roundness=None, lastCommittedPoint=None))
    doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": els,
           "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}
    (OUT / "system_design.excalidraw").write_text(json.dumps(doc, indent=1))


svg()
excalidraw()
print("wrote system_design.svg and system_design.excalidraw")

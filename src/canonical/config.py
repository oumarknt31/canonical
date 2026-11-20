"""Project-wide settings: paths, the three showcase anime, the eight CS topics, and model settings.

Secrets live only in .env (GEMINI_API_KEY). Nothing here should ever contain a key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CARDS_DIR = DATA / "cs_cards"
ANIME_DIR = DATA / "anime"
DB_PATH = DATA / "canonical.sqlite"
LABELS_CSV = DATA / "labels.csv"
MODELS_DIR = DATA / "models"
RESULTS_DIR = DATA / "results"

load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Leave empty to auto-pick the best available Flash model / embedding model for this key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "")
# Free-tier keys are rate limited per minute; we space calls to stay under it.
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "10"))

USER_AGENT = "canonical-student-project/0.1 (CUNY Tech Prep educational project; low-volume)"

# The three showcase anime, with their Fandom wiki subdomain and the canon pages to pull.
# Page names are search queries; the fetcher resolves each to the real wiki title.
ANIME = {
    "naruto": {
        "name": "Naruto",
        "wiki": "naruto",
        "pages": [
            "Chakra", "Shadow Clone Technique", "Multiple Shadow Clone Technique", "Rasengan",
            "Sharingan", "Byakugan", "Summoning Technique", "Sage Mode", "Tailed Beasts",
            "Jinchūriki", "Hand Seals", "Akatsuki", "Konohagakure", "Shinobi Ranks",
            "Chūnin Exams", "Summoning: Impure World Reincarnation", "Fūinjutsu", "Kage",
            "Naruto Uzumaki", "Sasuke Uchiha", "Kakashi Hatake", "Itachi Uchiha",
            "Fourth Shinobi World War", "Tsukuyomi", "Infinite Tsukuyomi", "Kamui",
            "Flying Thunder God Technique", "Eight Gates",
        ],
    },
    "hxh": {
        "name": "Hunter x Hunter",
        "wiki": "hunterxhunter",
        "pages": [
            "Nen", "Hatsu", "Water Divination", "Ten", "Zetsu", "Ren", "Gyo", "En",
            "Vows and Limitations", "Gon Freecss", "Killua Zoldyck", "Kurapika", "Hisoka Morow",
            "Chrollo Lucilfer", "Greed Island", "Hunter Exam", "Phantom Troupe",
            "Chimera Ant", "Isaac Netero", "Zoldyck Family", "Meruem", "Dark Continent", "Succession War",
            "Hunter Association", "Royal Guards", "Neferpitou", "Biscuit Krueger", "Ging Freecss", "Kite",
            "Heavens Arena", "Yorknew City arc", "Knuckle Bine", "Spell Cards", "Nen Exorcist", "Illumi Zoldyck",
        ],
    },
    "jjk": {
        "name": "Jujutsu Kaisen",
        "wiki": "jujutsu-kaisen",
        "pages": [
            "Cursed Energy", "Cursed Technique", "Domain Expansion", "Binding Vow", "Black Flash",
            "Reverse Cursed Technique", "Limitless", "Satoru Gojo", "Yuji Itadori",
            "Megumi Fushiguro", "Ten Shadows Technique", "Ryomen Sukuna", "Tokyo Jujutsu High",
            "Sorcerer Grades", "Shibuya Incident", "Culling Game", "Simple Domain",
            "Barrier Techniques", "Cursed Tools", "Kento Nanami", "Boogie Woogie",
            "Cursed Spirit Manipulation", "Suguru Geto", "Cursed Speech", "Idle Transfiguration",
            "Kenjaku", "Six Eyes",
        ],
    },
}

TOPICS = ["stack", "queue", "recursion", "bfs", "dfs", "hash_table", "binary_search", "deadlock"]


def load_card(topic: str) -> dict:
    return json.loads((CARDS_DIR / f"{topic}.json").read_text())


def load_cards() -> dict[str, dict]:
    return {t: load_card(t) for t in TOPICS if (CARDS_DIR / f"{t}.json").exists()}

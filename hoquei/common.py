import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DATA.mkdir(exist_ok=True)

USER_AGENT = "Mozilla/5.0 (compatible; hockey-stats-research/0.1)"

# Club prefixes/suffixes that differ between sources ("AD Valongo" vs "Valongo")
_NOISE = {
    "ad", "ud", "sc", "hc", "cd", "oc", "fc", "sl", "cp", "acr", "ha", "a", "ac", "ar", "ca",
    "cf", "gd", "gdr", "cr", "ard", "acd", "cdc", "ucd", "grd", "sad", "ss", "sr", "aa", "ae",
    "adc", "adr", "ahc", "ap", "as", "cs", "hcp", "rc", "ucs", "uds", "grc", "ao", "ac", "ass",
    "de", "da", "do", "das", "dos", "e", "clube", "hoquei", "patinagem",
}

# Manual aliases for keys that still differ after normalisation (normalised key -> canonical key)
ALIASES = {
    "juv viana": "juventude viana",
    "juventude de viana": "juventude viana",
    "paco arcos": "paco arcos",
    "fisica d": "fisica",
    "fisica torres vedras": "fisica",
    "candelaria": "candelaria",
    "acad espinho": "academica espinho",
    "academico espinho": "academica espinho",
    "acad feira": "academica feira",
    "academica de feira": "academica feira",
    "porto santo": "porto santo",
    "juv ouriense": "juventude ouriense",
    "s alenquer benf": "alenquer benfica",
    "alenquer e benfica": "alenquer benfica",
    # archive (2015-2021) name -> API (2021-) name
    "acad coimbra": "academica",
    "academica espinho": "espinho",
    "adj vila praia": "vila praia",
    "aj salesiana": "salesiana",
    "alenquer benfica": "alenquer",
    "biblioteca ir": "biblioteca",
    "c acad feira": "feira",
    "c infante sagres": "infante",
    "infante sagres": "infante",
    "ch carvalhos": "carvalhos",
    "crpf lavra": "lavra",
    "escola livre azemeis": "escola livre",
    "grf murches": "murches",
    "juventude pacense": "pacense",
    "os limianos": "limianos",
    "os tigres": "tigres",
    "riba ave": "riba dave",
    "riba d ave": "riba dave",
    "usc paredes": "paredes",
    "hcp grandola": "grandola",
    "valenca": "valenca",
    # 2007-2015 archive names
    "escola livre az": "escola livre",
    "juv pacense": "pacense",
    "uf entroncamento": "entroncamento",
    "fund nortecoope": "nortecoope",
}

# Pretty display name for each canonical key (filled lazily: first name seen wins)
DISPLAY = {}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def team_key(name):
    if name is None or (isinstance(name, float)):
        return None
    s = _strip_accents(str(name)).lower()
    s = re.sub(r"&#8217;|['’`\".]", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = [w for w in s.split() if w not in _NOISE]
    key = " ".join(words) or s.strip()
    for _ in range(3):  # aliases can chain (acad espinho -> academica espinho -> espinho)
        key = ALIASES.get(key, key)
    return key


def canonical_team(name):
    key = team_key(name)
    if key is not None and key not in DISPLAY:
        DISPLAY[key] = str(name)
    return key

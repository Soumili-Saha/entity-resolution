"""Text cleaning for names and addresses. Country-agnostic, hand-written rules only.

Noise seen in the training data that these rules target:
  names    : legal suffixes anywhere ("Private Anand Foundation Ltd"), junk marks ("***", "<<",
             "(ID: 54473)", "[Ltd]"), domain-style names ("UROLOGYSTRATEGICHEALTH.COM",
             "@gibsongallardo"), aliases ("X fka Y", "X a/k/a Y", "X | www.y.com"), digit/letter
             OCR swaps ("N0SE", "6ULF"), Devanagari/Bengali transliterations of English names.
  addresses: abbreviations both ways (Rd/Road, St/Street/"SAINT"), reordered components,
             state codes vs names vs native script, missing parts.

Canonical forms are the SHORT ones (road -> rd, street -> st) so that both directions agree.
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from unidecode import unidecode

from src.cpus import n_cpus

# Legal / corporate designators and filler words (multilingual).
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "ltda", "gmbh", "opc", "pllc", "pc", "pa",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "cie", "the", "and", "of",
    "le", "la", "les", "de", "du", "des", "et", "l", "d", "ms", "m/s", "mrs", "sri", "shri",
}

NAME_CANON = {
    "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "services",
    "bros": "brothers", "assoc": "associates", "assocs": "associates", "ent": "enterprises",
    "enterprise": "enterprises", "tech": "technologies", "technology": "technologies",
    "sys": "systems", "mgmt": "management", "natl": "national", "ctr": "center", "centre": "center",
    "grp": "group", "inds": "industries", "industry": "industries", "solution": "solutions",
    "service": "services", "product": "products", "st": "saint", "ste": "sainte", "mt": "mount",
    "ft": "fort", "dr": "doctor", "hosp": "hospital", "univ": "university",
}

ADDR_CANON = {
    "road": "rd", "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "lane": "ln", "drive": "dr", "court": "ct",
    "place": "pl", "square": "sq", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "trail": "trl", "suite": "ste", "floor": "fl", "flr": "fl",
    "apartment": "apt", "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "near": "nr", "opposite": "opp", "market": "mkt", "nagar": "ngr", "sector": "sec",
    "extension": "extn", "ext": "extn", "number": "no", "num": "no", "district": "dist",
    "chemin": "ch", "che": "ch", "impasse": "imp", "allee": "all", "faubourg": "fg", "fbg": "fg",
    "route": "rte", "rue": "r", "quai": "qu", "cite": "cte", "residence": "res", "resid": "res",
    "batiment": "bat", "bt": "bat", "lieu": "", "dit": "", "cedex": "", "cross": "crs", "main": "mn", "unit": "ste", "house": "hse", "city": "",
    "town": "", "cp": "", "null": "", "po": "", "box": "", "first": "1st", "second": "2nd",
    "third": "3rd", "fourth": "4th",
}

LANDMARK = {"nr", "opp", "behind", "beside", "next", "adjacent", "pres", "face", "landmark"}

_ALIAS = re.compile(r"\b(?:fka|f/k/a|aka|a/k/a|dba|d/b/a|t/a|formerly|trading as)\b", re.I)
_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|co|fr|us|biz|info|io)\b", re.I)
_IDTAG = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_DIGIT_IN_WORD = re.compile(r"(?<=[a-z])[0136](?=[a-z])|^[0136](?=[a-z]{2})")
_OCR = str.maketrans({"0": "o", "1": "i", "3": "e", "6": "g"})
_VOWELS = re.compile(r"(?<=.)[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")
_POSTAL = re.compile(r"\b(\d{3}\s?\d{3}|\d{5})\b")
_NUM = re.compile(r"\d+")


def _basic(text):
    """Transliterate to ASCII, lowercase, '&' -> 'and', merge dotted initials, drop punctuation."""
    t = unidecode(text or "").lower().replace("&", " and ").replace("'", "")
    t = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", t)
    t = _NON_ALNUM.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def _join_initials(tokens):
    """['a','b','c','foods'] -> ['abc','foods'] (initials written with spaces or dots)."""
    out, buf = [], []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            buf.append(tok)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(tok)
    if buf:
        out.append("".join(buf))
    return out


_PHONETIC = [("bh", "v"), ("w", "v"), ("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"),
             ("kh", "k"), ("gh", "g"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("z", "j"), ("x", "ks")]


def skeleton(token):
    """Phonetic consonant skeleton: fold sound-alike letters, keep first letter, drop later
    vowels, squeeze repeats. 'limittedd' / 'limited' -> 'lmtd', 'sebhen' / 'seven' -> 'svn'."""
    for a, b in _PHONETIC:
        token = token.replace(a, b)
    return _REPEAT.sub(r"\1", _VOWELS.sub("", token))


# Skeletons of legal words, so transliterated ones ("praaivett limittedd") are dropped too.
LEGAL_SKEL = {skeleton(w) for w in ("private", "limited", "incorporated", "corporation", "company", "pvt", "ltd", "llp")} | {"elp"}


def _fix_ocr(tok):
    """Digits inside words are OCR noise in names: 'n0se' -> 'nose', '6ulf' -> 'gulf'."""
    if tok.isdigit() or not any(c.isdigit() for c in tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    return tok.translate(_OCR) if letters >= 2 else tok


def name_views(raw):
    """All name views for one raw name: (clean, core, skel, nospace, alt).
    clean = cleaned full name, core = legal words removed, skel = consonant skeleton of core,
    nospace = core without spaces (for domain-style names), alt = alias part (fka/aka/| url)."""
    text = _IDTAG.sub(" ", raw or "")
    alt = ""
    parts = re.split(r"\s\|\s", text, maxsplit=1)
    if len(parts) == 2:
        text, alt = parts
    m = _ALIAS.search(text)
    if m:
        text, alt2 = text[:m.start()], text[m.end():]
        if len(_basic(alt2)) > 0:
            text, alt = alt2, text  # the name after 'fka' is usually the S1 name; keep both
    text = _URL.sub(r" \1 ", text)
    alt = _URL.sub(r" \1 ", alt)
    toks = _join_initials([_fix_ocr(t) for t in _basic(text).split()])
    toks = [NAME_CANON.get(t, t) for t in toks]
    clean = " ".join(toks)
    core_toks = [t for t in toks if t not in LEGAL and skeleton(t) not in LEGAL_SKEL]
    core = " ".join(core_toks) or clean
    skel = " ".join(skeleton(t) for t in core.split())
    alt_clean = " ".join(t for t in _basic(alt).split() if t not in LEGAL)
    return clean, core, skel, core.replace(" ", ""), alt_clean


def address_views(raw):
    """Address views: (clean, postal codes, numbers, landmark tokens), all as space-joined strings."""
    raw = raw or ""
    ascii_raw = unidecode(raw)
    toks = []
    for t in _basic(raw).split():
        t = ADDR_CANON.get(t, t)
        if t:
            toks.append(t)
    clean = " ".join(toks)
    postal = " ".join(sorted({p.replace(" ", "") for p in _POSTAL.findall(ascii_raw)}))
    nums = " ".join(sorted(set(_NUM.findall(ascii_raw))))
    lm = set()
    for i, t in enumerate(toks):
        if t in LANDMARK:
            lm.update(toks[i + 1:i + 4])
    return clean, postal, nums, " ".join(sorted(lm - LANDMARK))


def _normalise_chunk(args):
    """Worker: normalise lists of names and addresses."""
    names, addrs = args
    n = [name_views(x) for x in names]
    a = [address_views(x) for x in addrs]
    return n, a


def normalize_frame(df, workers=None):
    """Raw source DataFrame -> same frame with cleaned columns added:
    name_clean, name_core, name_skel, name_ns, name_alt, addr_clean, postal, nums, landmarks, country_norm."""
    df = df.copy()
    for c in ("business_name", "business_address", "country"):
        df[c] = df[c].fillna("").astype(str)
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    workers = workers or n_cpus()
    step = max(20000, len(df) // (workers * 4) + 1)
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(df), step)]
    nv, av = [], []
    if len(chunks) == 1 or workers == 1:
        results = map(_normalise_chunk, chunks)
    else:
        results = ProcessPoolExecutor(workers).map(_normalise_chunk, chunks)
    for n, a in results:
        nv.extend(n)
        av.extend(a)
    ncols = ["name_clean", "name_core", "name_skel", "name_ns", "name_alt"]
    acols = ["addr_clean", "postal", "nums", "landmarks"]
    for i, c in enumerate(ncols):
        df[c] = [v[i] for v in nv]
    for i, c in enumerate(acols):
        df[c] = [v[i] for v in av]
    df["name_alt_skel"] = [" ".join(skeleton(t) for t in s.split()) for s in df["name_alt"]]
    df["country_norm"] = df["country"].str.strip().str.lower()
    return df


def token_set(s):
    """Space-joined string -> set of tokens (empty string -> empty set)."""
    return set(s.split()) if s else set()


def acronym(core):
    """Initials of a multi-word core name ('tata consultancy services' -> 'tcs')."""
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


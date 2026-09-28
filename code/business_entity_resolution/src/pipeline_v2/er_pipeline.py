"""
er_pipeline.py - Business Entity Resolution pipeline (ML Challenge 2026)

Stages
  1. prepare_split     : clean names (incl. Indic), parse addresses, per source, cached parquet
  2. detect_fillers    : words injected into S2/S3 far more often than S1 (label-free)
  3. assign_partitions : country -> state (if states are detectable) else one block per country;
                         records without usable address go to the country's no-address pool
  4. run_blocking      : exact keys + TF-IDF nearest neighbours, ranked, top-K per S1
  5. build_features    : similarity + competition features per candidate pair
  6. cross_validate    : LightGBM, grouped folds by S1 (out-of-fold probabilities)
  7. evaluate          : one-S1-per-S2/S3 rule, threshold tuned for macro F0.5 vs ground truth

Only label-free statistics are computed from the data being processed; the only artefact
learned from train labels is the Indic->English dictionary (indic_dict.json).

v2.2 changes (both additive - old cached prep/selection/feature parquet files still load; a model
trained on v2.1's FEATURES_S1/FEATURES_S2 still works, it just won't have the new columns):
  1. Legal-suffix family features (legal_a_present, legal_b_present, legal_changed,
     legal_changed_no_addr, legal_asym_no_addr) - targets the false-merge pattern from the v2.1
     error log where the name matches and the S2/S3 side has no address to confirm identity, but the
     legal suffix changed to a genuinely different family (Ltd->Corp, PC->Center Inc, Limited->LLP)
     or appeared/disappeared entirely (Sam, English & Kerley -> ...INC). Deliberately does NOT fire on
     the routine within-family swaps the EDA found in true matches (pc->inc, corp->corporation,
     limited->ltd - see _LEGAL_FAMILY). New PREP_COLS field "legal" (old cached prep parquet files
     lack it; load_selection falls back to "" for those via the has-column check in _setup_features).
  2. parse_addr's Indic tokens now go through the same resolve() (dictionary -> skeleton fallback ->
     rules) pipeline that name_tokens already uses, instead of raw translit() rules only. Addresses
     never had dictionary coverage before; this closes that gap (e.g. Telugu/Devanagari address words
     like "flat", "chambers" now normalise the same way whichever side of a pair they appear on).

v2.3 changes (additive - old cached prep/selection/blocks/features/features_extra parquet files still
load unchanged; only oof_s1.parquet, sib.npz and oof_s2.parquet need deleting+retraining, since
FEATURES_S1 grows one column):
  1. One semantic-embedding-similarity feature (emb_cos, section 11 below). A multilingual sentence
     encoder (default: intfloat/multilingual-e5-small, MIT licensed, 118M params) embeds each record's
     "name | address" text once (GPU pass, cached to embed/emb.npy, aligned with S's row order); emb_cos
     is the cosine similarity between a candidate pair's two vectors. Targets the class of stage-1
     errors the hand-written features keep missing - word-order shuffles, dropped words, glued/reordered
     names, and unseen Indic / non-English vocabulary not covered by indic_dict.json - without hand-writing
     another rule per pattern. Deliberately NOT used for blocking (the EDA's Step 14 conclusion holds:
     no_addr / name_dead candidates have no text for an embedding to find) - stage-1 model feature only,
     computed over the SAME candidates blocking already produced.
  (v2.3's embedding feature is REMOVED in v3.0 - it moved final F0.5 by -0.00002 and cost ~20 min GPU.)

v3.0 changes (CPU only; prep/, selected.parquet, blocks/, features/, features_extra/ caches all reused;
stage-1/stage-2 models retrain because FEATURES_S1 / SIB_FEATURES changed):
  A. realistic_mask(): training/validation drop S2/S3 records whose true S1 is OUTSIDE the selection
     ("orphans"). With 15% of S1 but the whole country's no-address pool, orphans looked like strong
     matches to a wrong S1 (r2=0, identical name) and taught the model "identical name + no address =
     unreliable". Test has no orphans. Labels are used only to pick training ROWS, never as features.
     cv_train(): early stopping (inner group split), fold models kept; monotone method "intermediate".
  B. expand_candidates(): after stage 1, each S1's confident matches (and the S1 itself) seed exact
     name / glued-name / full-address keys; S2/S3 records sharing a (non-generic) key become new
     candidates. Raises the blocking ceiling for no-address duplicates and random-name records.
  C. FEATURES_V3 (22 cols): synthetic-name detector (label-free syllable inventory), glued-name
     splitting against the S1 vocabulary, exact-address / number-set features, private-vs-public /
     legal-set conflicts, name-duplicate counts (how many S1s carry exactly this name), expansion flags.
     SIB_FEATURES +3: sib_tsr_max, sib_addr_exact, b_conf_elsewhere.
  D. expected_f_keep(): per-S1 choice of the match set that maximises EXPECTED F0.5 (singletons
     included), tuned on validation against the old global threshold; the better one is used.
"""
import os, re, csv, json, time, gc, math, unicodedata
from functools import lru_cache
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

VERSION = "3.0"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# =====================================================================================
# 1. TEXT BASICS
# =====================================================================================
INDIC_RE = re.compile(r"[ऀ-෿]")


def base(s):
    """Unicode-normalise, drop zero-width joiners, lowercase, strip Latin accents (keeps Indic)."""
    s = unicodedata.normalize("NFKC", s or "").lower().replace("‌", "").replace("‍", "")
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[̀-ͯ]", "", s)


def has_indic(s):
    return bool(INDIC_RE.search(s or ""))


def stem(x):
    if len(x) > 4 and x.endswith("ies"):
        return x[:-3] + "y"
    if len(x) > 3 and x.endswith("s") and not x.endswith("ss"):
        return x[:-1]
    return x


_FOLD = str.maketrans("gl", "bi")


def fold(x):
    """Merge OCR look-alike letters (used only for glued-name keys)."""
    return x.replace("rn", "m").replace("cl", "d").replace("vv", "w").translate(_FOLD)


# =====================================================================================
# 2. NAME NORMALISATION
# =====================================================================================
LEGAL = set("""llc inc incorporated corp corporation co company cos ltd limited pvt private llp lp lllp
pc pllc plc pa psc gmbh ag sa sarl sas sasu eurl sci snc""".split())
HONOR = set("mr mrs ms smt shri sri dr the a an of and".split())
_INDIC_LEGAL_RAW = """लिमिटेड प्राइवेट प्रा लि एलएलपी లిమిటెడ్ ప్రైవేట్ ಲಿಮಿಟೆಡ್ ಪ್ರೈವೇಟ್
லிமிடெட் பிரைவேட் লিমিটেড প্রাইভেট લિમિટેડ પ્રાઇવેટ ലിമിറ്റഡ് പ്രൈവറ്റ് ଲିମିଟେଡ୍ ପ୍ରାଇଭେଟ୍"""
INDIC_LEGAL = {base(w) for w in _INDIC_LEGAL_RAW.split()}
FILLER = {"center", "centre", "services", "service", "india", "partners", "partner"}
CRED = set("md do dds dmd phd cpa esq rn dc od dpm dvm lcsw".split())
DROP_FINAL = FILLER | CRED | LEGAL | HONOR

ID_RE = re.compile(r"\(?\bid\s*[:#]?\s*\d+\s*\)?")
DOM_RE = re.compile(r"\bwww\.|\.(com|net|org|in|co|biz|info|us|fr|eu)\b")
MS_RE = re.compile(r"\bm\s*/\s*s\b")
MARK_RE = re.compile(r"\b(d\s*/?\s*b\s*/?\s*a|a\s*/?\s*k\s*/?\s*a|f\s*/?\s*k\s*/?\s*a|nee|t\s*/\s*a|"
                     r"formerly(\s+known\s+as)?|doing\s+business\s+as|also\s+known\s+as|known\s+as|trading\s+as)\b")
DOT_RE = re.compile(r"\b(?:[a-z]\s*\.\s*){2,}(?:[a-z]\b)?")
_OCR = str.maketrans("01568", "olsbg")
_PUNCT_RE = re.compile(r"[^\w\sऀ-෿]|_")
DOMAIN_FLAG_RE = re.compile(r"(?i)\.(com|net|org|in|co|biz|info|us|fr|eu)\b|www\.|^\s*[@#]")


def name_tokens_v1(s):
    """Clean name -> tokens (legal forms, honorifics, markers, domains removed; OCR digits fixed)."""
    s = base(s)
    s = ID_RE.sub(" ", s); s = DOM_RE.sub(" ", s); s = MS_RE.sub(" ", s); s = MARK_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " ")
    s = DOT_RE.sub(lambda m: " " + re.sub(r"[^a-z]", "", m.group(0)) + " ", s)
    s = _PUNCT_RE.sub(" ", s)
    t = [x.translate(_OCR) if (re.search(r"[a-z]", x) and re.search(r"\d", x)) else x for x in s.split()]
    out, run = [], []
    for x in t + [""]:                                   # "l l c" -> "llc"
        if len(x) == 1 and x.isalpha():
            run.append(x); continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0]); run = []
        if x:
            out.append(x)
    drop = LEGAL | HONOR | INDIC_LEGAL | {"com", "www"}
    return [x for x in out if x not in drop]


# legal-suffix family map: groups suffixes that swap harmlessly within true matches (per EDA step 4:
# pc->inc, lp->llc, corp->corporation, limited->ltd, etc. are NOT signals of a different business),
# so legal_changed below only fires on a real cross-family swap, not routine abbreviation variance.
_LEGAL_FAMILY = {}
for _fam, _members in [
    ("corp", ["corp", "corporation"]),
    ("inc", ["inc", "incorporated"]),
    ("ltd", ["ltd", "limited"]),
    ("pc", ["pc", "pllc", "plc", "pa", "psc"]),
    ("llc", ["llc"]),
    ("llp", ["llp", "lp", "lllp"]),
    ("co", ["co", "company", "cos"]),
    ("other", ["pvt", "private", "gmbh", "ag", "sa", "sarl", "sas", "sasu", "eurl", "sci", "snc"]),
]:
    for _m in _members:
        _LEGAL_FAMILY[_m] = _fam
del _fam, _members, _m


def _raw_name_tokens(s):
    """name_tokens_v1's cleaning WITHOUT the legal/honorific drop (shared by legal_suffix_of/legal_set_of)."""
    s = base(s or "")
    s = ID_RE.sub(" ", s); s = DOM_RE.sub(" ", s); s = MS_RE.sub(" ", s); s = MARK_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " ")
    s = DOT_RE.sub(lambda m: " " + re.sub(r"[^a-z]", "", m.group(0)) + " ", s)
    s = _PUNCT_RE.sub(" ", s)
    t = [x.translate(_OCR) if (re.search(r"[a-z]", x) and re.search(r"\d", x)) else x for x in s.split()]
    out, run = [], []
    for x in t + [""]:
        if len(x) == 1 and x.isalpha():
            run.append(x); continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0]); run = []
        if x:
            out.append(x)
    return out


# v3.0: order-free canonical legal SET (Private Limited != Limited != Public Limited in India)
_INDIC_PRIV = {base(w) for w in "प्राइवेट प्रा ప్రైవేట్ ಪ್ರೈವೇಟ್ பிரைவேட் প্রাইভেট પ્રાઇવેટ പ്രൈവറ്റ് ପ୍ରାଇଭେଟ୍".split()}
_INDIC_LLP = {base("एलएलपी")}
_CANON = {"pvt": "private", "private": "private", "ltd": "limited", "limited": "limited",
          "corp": "corporation", "corporation": "corporation", "inc": "incorporated",
          "incorporated": "incorporated", "co": "company", "company": "company", "cos": "company"}


def legal_set_of(s):
    out = set()
    for x in _raw_name_tokens(s):
        if x in LEGAL:
            out.add(_CANON.get(x, x))
        elif x in _INDIC_PRIV:
            out.add("private")
        elif x in _INDIC_LLP:
            out.add("llp")
        elif x in INDIC_LEGAL:
            out.add("limited")
        elif x == "public":
            out.add("public")
    return " ".join(sorted(out))


def legal_suffix_of(s):
    """Returns the coarse legal-suffix FAMILY present in a raw business name, or '' if none.
    Reuses name_tokens_v1's own cleaning (dotted-initial joining, OCR digit fix, id/domain/marker
    stripping) so 'L.L.C.' / 'lnc' / etc. are recognised the same way name_tokens_v1 recognises them
    before dropping them - this just runs before the drop instead of after."""
    s = base(s or "")
    s = ID_RE.sub(" ", s); s = DOM_RE.sub(" ", s); s = MS_RE.sub(" ", s); s = MARK_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " ")
    s = DOT_RE.sub(lambda m: " " + re.sub(r"[^a-z]", "", m.group(0)) + " ", s)
    s = _PUNCT_RE.sub(" ", s)
    t = [x.translate(_OCR) if (re.search(r"[a-z]", x) and re.search(r"\d", x)) else x for x in s.split()]
    out, run = [], []
    for x in t + [""]:
        if len(x) == 1 and x.isalpha():
            run.append(x); continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0]); run = []
        if x:
            out.append(x)
    for x in reversed(out):                              # legal suffix is almost always at the end
        if x in LEGAL:
            return _LEGAL_FAMILY.get(x, "other")
    return ""


# =====================================================================================
# 3. INDIC SCRIPT HANDLING (dictionary -> legal drop -> conservative lookup -> transliteration)
# =====================================================================================
_SCRIPTS = None


def _scripts():
    global _SCRIPTS
    if _SCRIPTS is None:
        from indic_transliteration import sanscript as S
        _SCRIPTS = [(0x0900, 0x097F, S.DEVANAGARI), (0x0980, 0x09FF, S.BENGALI),
                    (0x0A00, 0x0A7F, S.GURMUKHI), (0x0A80, 0x0AFF, S.GUJARATI),
                    (0x0B00, 0x0B7F, S.ORIYA), (0x0B80, 0x0BFF, S.TAMIL),
                    (0x0C00, 0x0C7F, S.TELUGU), (0x0C80, 0x0CFF, S.KANNADA),
                    (0x0D00, 0x0D7F, S.MALAYALAM)]
    return _SCRIPTS


def script_of(w):
    for ch in w:
        o = ord(ch)
        for lo, hi, sc in _scripts():
            if lo <= o <= hi:
                return sc
    return None


_PRE3 = str.maketrans({"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े", "ઑ": "ઓ", "ૉ": "ો",
                       "ൽ": "ല്", "ർ": "ര്", "ൻ": "ന്", "ൺ": "ണ്", "ൾ": "ള്", "ൿ": "ക്"})


@lru_cache(maxsize=1_000_000)
def translit(w):
    sc = script_of(w)
    if sc is None:
        return w
    from indic_transliteration import sanscript as S
    x = S.transliterate(unicodedata.normalize("NFC", w).translate(_PRE3), sc, S.IAST)
    x = x.replace("ṃ", "n").replace("ṁ", "n").replace("ḥ", "").replace("ṛ", "ri")
    x = re.sub(r"[^a-z]", "", base(x))
    if sc == S.TAMIL:                                   # Tamil has no aspirates
        x = x.replace("jh", "s")
        x = re.sub(r"([kgcjtdpb])h", r"\1", x)
    if len(x) > 3 and x.endswith("a") and x[-2] not in "aeiou":
        x = x[:-1]                                      # schwa deletion
    return re.sub(r"j$", "s", x)                        # final ज = plural s


_VOW = "aeiou"


@lru_cache(maxsize=1_000_000)
def skel(w, voice=False):
    """Consonant sound skeleton, e.g. consultancy -> knsltns."""
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return ""
    w = re.sub(r"[ts]ion", "sn", w)
    w = re.sub(r"t(?=ur)", "c", w)
    w = re.sub(r"c(?=[eiy])", "s", w); w = re.sub(r"g(?=[eiy])", "j", w)
    for a, b in [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
                 ("ch", "c"), ("jh", "j"), ("ck", "k"), ("w", "v"), ("q", "k"), ("x", "ks"), ("z", "j"),
                 ("c", "k"), ("y", "i")]:
        w = w.replace(a, b)
    if voice:
        w = w.translate(str.maketrans("dbg", "tpk"))
    out = ("_" if w[0] in _VOW else w[0]) + "".join(ch for ch in w[1:] if ch not in _VOW)
    out = re.sub(r"(.)\1+", r"\1", out)
    return out[:-1] if len(out) > 2 and out.endswith("s") else out


_LEGAL_SK = None


def is_legal_tl(tl):
    global _LEGAL_SK
    if _LEGAL_SK is None:
        _LEGAL_SK = {skel(x) for x in ["limited", "private", "pvt", "ltd", "llp"]}
    return (tl in {"pra", "li", "pi", "el"}
            or (len(tl) <= 12 and "l" in tl and set(tl) <= set("aeilpbh"))
            or skel(tl) in _LEGAL_SK)


# resources: dictionary learned from train labels + vocabulary index built from S1 names (label-free)
_IND = {"dict": {}, "sk": {}, "skv": {}}


def build_vocab_index(voc):
    """voc: Counter(word -> count) of stemmed S1 name tokens. Returns (sk, skv) candidate indexes."""
    sk, skv = defaultdict(list), defaultdict(list)
    for t, c in voc.items():
        if has_indic(t) or not t.isalpha():
            continue
        sk[skel(t)].append((t, c)); skv[skel(t, True)].append((t, c))
    for d in (sk, skv):
        for k in d:
            d[k] = sorted(d[k], key=lambda z: -z[1])[:8]
    return dict(sk), dict(skv)


def set_indic_resources(dict_map, sk, skv):
    _IND["dict"] = dict_map or {}; _IND["sk"] = sk or {}; _IND["skv"] = skv or {}
    resolve.cache_clear()


def _pick(tl, cands):
    """Conservative: only frequent words, only when spelling is clearly close."""
    from rapidfuzz.fuzz import ratio
    cands = [z for z in (cands or []) if z[1] >= 20]
    if not cands:
        return None
    w = max(cands, key=lambda z: ratio(tl, z[0]) + 10 * math.log10(z[1]))[0]
    return w if ratio(tl, w) >= 75 else None


@lru_cache(maxsize=1_000_000)
def resolve(t):
    if not has_indic(t):
        return t
    d = _IND["dict"]
    if t in d:
        return d[t]
    tl = translit(t)
    if is_legal_tl(tl):
        return None
    return _pick(tl, _IND["sk"].get(skel(tl))) or _pick(tl, _IND["skv"].get(skel(tl, True))) or tl


def name_tokens(s):
    out = []
    for t in name_tokens_v1((s or "").replace("ॐ", " om ")):
        r = resolve(t)
        if r is None:
            continue
        r = stem(r)
        if r in DROP_FINAL or (len(r) == 1 and not r.isdigit()):
            continue
        out.append(r)
    return out


# =====================================================================================
# 4. ADDRESS PARSING
# =====================================================================================
US_ST = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
         "colorado": "co", "connecticut": "ct", "delaware": "de", "districtofcolumbia": "dc", "florida": "fl",
         "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
         "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
         "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
         "montana": "mt", "nebraska": "ne", "nevada": "nv", "newhampshire": "nh", "newjersey": "nj",
         "newmexico": "nm", "newyork": "ny", "northcarolina": "nc", "northdakota": "nd", "ohio": "oh",
         "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhodeisland": "ri", "southcarolina": "sc",
         "southdakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
         "virginia": "va", "washington": "wa", "westvirginia": "wv", "wisconsin": "wi", "wyoming": "wy"}
US_CODES = set(US_ST.values())

IN_ST = {
    "mh": (["maharashtra"], ["महाराष्ट्र"]), "up": (["uttar pradesh"], ["उत्तर प्रदेश"]),
    "wb": (["west bengal"], ["পশ্চিমবঙ্গ"]), "gj": (["gujarat"], ["ગુજરાત"]),
    "tn": (["tamil nadu"], ["தமிழ்நாடு"]), "ka": (["karnataka"], ["ಕರ್ನಾಟಕ"]),
    "aptg": (["telangana", "andhra pradesh"], ["తెలంగాణ", "ఆంధ్రప్రదేశ్"]),
    "hr": (["haryana"], ["हरियाणा"]), "rj": (["rajasthan"], ["राजस्थान"]),
    "br": (["bihar"], ["बिहार"]), "kl": (["kerala", "keralam"], ["കേരളം"]),
    "mp": (["madhya pradesh"], ["मध्य प्रदेश"]), "pb": (["punjab"], ["ਪੰਜਾਬ"]),
    "od": (["odisha", "orissa"], ["ଓଡ଼ିଶା"]), "dl": (["delhi"], ["दिल्ली"]),
    "as": (["assam"], ["অসম"]), "jh": (["jharkhand"], ["झारखंड"]),
    "cg": (["chhattisgarh"], ["छत्तीसगढ़"]), "uk": (["uttarakhand"], ["उत्तराखंड"]),
    "hp": (["himachal pradesh"], ["हिमाचल प्रदेश"]), "ga": (["goa"], ["गोवा"]),
    "jk": (["jammu and kashmir", "jammu kashmir"], []), "ch": (["chandigarh"], []),
    "py": (["puducherry", "pondicherry"], []),
}
IN_CODE_MAP = {"mh": "mh", "up": "up", "wb": "wb", "gj": "gj", "tn": "tn", "ka": "ka", "tg": "aptg", "ap": "aptg",
               "ts": "aptg", "hr": "hr", "rj": "rj", "br": "br", "kl": "kl", "mp": "mp", "pb": "pb", "od": "od",
               "or": "od", "dl": "dl", "as": "as", "jh": "jh", "cg": "cg", "uk": "uk", "hp": "hp", "ga": "ga",
               "jk": "jk", "ch": "ch", "py": "py"}


def _letters(toks):
    return re.sub(r"[^a-z]", "", "".join(toks))


_IN_EXACT, _IN_SK = None, None


def _india_maps():
    global _IN_EXACT, _IN_SK
    if _IN_SK is None:
        ex, sk = {}, {}
        for code, (eng, nat) in IN_ST.items():
            for n in eng:
                ex[_letters(n.split())] = code
                sk[skel(_letters(n.split()), True)] = code
            for n in nat:
                sk[skel(_letters([translit(t) for t in base(n).split()]), True)] = code
        _IN_EXACT, _IN_SK = ex, sk
    return _IN_EXACT, _IN_SK


ABBR = {"street": "st", "str": "st", "saint": "st", "road": "rd", "drive": "dr", "drv": "dr", "avenue": "av",
        "ave": "av", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "boulevard": "bd",
        "blvd": "bd", "trail": "trl", "terrace": "ter", "highway": "hwy", "parkway": "pkwy", "square": "sq",
        "rue": "r", "impasse": "imp", "allee": "all", "chemin": "ch", "chem": "ch", "route": "rte",
        "sainte": "ste", "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft",
        "point": "pt", "heights": "hts", "junction": "jct", "quai": "q", "cours": "crs"}
UNIT = {"unit", "apt", "apartment", "appt", "appartement", "suite", "flat", "room", "rm", "bldg", "building",
        "apto"}
FLOOR = {"floor", "flr", "flor", "etage", "fl"}
JUNKN = {"po", "box", "pmb", "bp", "cs", "cedex"}
PREFIX = {"no", "h", "hno", "house", "door", "plot", "shop", "sr", "survey", "s", "hn"}
ASTOP = {"de", "la", "le", "les", "du", "des", "d", "l", "of", "the", "and", "city", "town", "village",
         "township", "cdp", "county", "region", "corporation", "null", "na", "nan", "none"}
_ORD1 = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth "
                                          "eleventh twelfth thirteenth fourteenth fifteenth sixteenth "
                                          "seventeenth eighteenth nineteenth".split())}
_ONES = {**{w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}, **_ORD1}
_TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_TENS.update({w: (i + 2) * 10 for i, w in enumerate(
    "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split())})
_DIGIT_RE = re.compile(r"\d")
_ORDSUF_RE = re.compile(r"^(\d+)(st|nd|rd|th|er|e|eme|re)$")


def state_of(toks, ctry):
    j = _letters(toks)
    if not j or any(_DIGIT_RE.search(t) for t in toks):
        return None
    if ctry == "US":
        if j in US_ST:
            return US_ST[j]
        if len(toks) == 1 and toks[0] in US_CODES:
            return toks[0]
    elif ctry == "India":
        if len(toks) == 1 and toks[0] in IN_CODE_MAP:
            return IN_CODE_MAP[toks[0]]
        ex, sk = _india_maps()
        if j in ex:
            return ex[j]
        k = skel(j, True)
        if len(k) >= 3:
            return sk.get(k)
    return None


def _parse_tokens(toks):
    out, i = [], 0
    while i < len(toks):                               # number words -> digits
        t = toks[i]
        if t in _TENS and i + 1 < len(toks) and toks[i + 1] in _ONES and _ONES[toks[i + 1]] < 10:
            out.append(str(_TENS[t] + _ONES[toks[i + 1]])); i += 2; continue
        out.append(str(_TENS[t]) if t in _TENS else str(_ORD1[t]) if t in _ORD1 else t); i += 1
    words, nums, units, prev_num, j = [], [], [], False, 0
    while j < len(out):
        t = _ORDSUF_RE.sub(r"\1", out[j])
        nxt = out[j + 1] if j + 1 < len(out) else ""
        if t in UNIT:
            if nxt:
                units.append(nxt); j += 2
            else:
                j += 1
            prev_num = False; continue
        if t in FLOOR:
            if prev_num and nums:
                units.append(nums.pop())
            j += 1; prev_num = False; continue
        if t in JUNKN:
            j += 1
            while j < len(out) and (out[j] in JUNKN or _DIGIT_RE.search(out[j])):
                j += 1
            prev_num = False; continue
        if t in PREFIX and (_DIGIT_RE.search(nxt) or nxt in PREFIX):
            j += 1; continue
        if t in {"bis", "ter", "b"} and prev_num:
            j += 1; continue
        if _DIGIT_RE.search(t):
            nums.extend(str(int(d)) for d in re.findall(r"\d+", t)); prev_num = True
        else:
            prev_num = False
            t = ABBR.get(t, t)
            if t not in ASTOP:
                words.append(t)
        j += 1
    return words, nums, units


def parse_addr(s, ctry):
    s = re.sub(r"(\d)\s*-\s*(\d)", r"\1\2", s or "")
    s = base(s)
    s = re.sub(r"\bn\s*[°º]\s*", " no ", s).replace("#", " no ")
    state, words, nums, units = None, [], [], []
    for c in s.split(","):
        toks = _PUNCT_RE.sub(" ", c).split()
        toks = [(resolve(t) or translit(t)) if has_indic(t) else t for t in toks]
        if not toks or _letters(toks) in {"null", "na", "none", "nan"}:
            continue
        st = state_of(toks, ctry)
        if st:
            state = st; continue
        w, n, u = _parse_tokens(toks)
        words += w; nums += n; units += u
    return state, words, nums, units


# =====================================================================================
# 5. STAGE 1 - PREPARE
# =====================================================================================
READ_KW = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False,
               encoding="utf-8", on_bad_lines="warn")
PREP_COLS = ["entity_id", "src", "country", "state", "toks", "words", "nums", "units",
             "has_addr", "addr_ok", "name_indic", "name_domain", "legal"]


def load_raw(split, k, data_dir, parquet_dir=None):
    """Raw source file k (1/2/3) of split ('train'/'test'); uses the parquet copy if available."""
    if parquet_dir:
        for fn in ([f"source{k}.parquet", f"train_source{k}.parquet"] if split == "train"
                   else [f"test_source{k}.parquet"]):
            p = os.path.join(parquet_dir, fn)
            if os.path.exists(p):
                df = pd.read_parquet(p)
                return df.astype(str).replace({"<NA>": "", "nan": "", "None": ""})
    return pd.read_csv(os.path.join(data_dir, split, f"{split}_source{k}.tsv"), **READ_KW)


def load_ground_truth(data_dir, parquet_dir=None):
    """Returns DataFrame(s1_id, m_id) with one row per true pair."""
    if parquet_dir and os.path.exists(os.path.join(parquet_dir, "pairs.parquet")):
        p = pd.read_parquet(os.path.join(parquet_dir, "pairs.parquet"))
        return p[["s1_id", "m_id"]].astype(str)
    gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), **READ_KW)
    gt["m_id"] = gt.matched_entity_ids.str.split(",")
    p = gt.explode("m_id").rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "m_id"]]
    p = p[p.m_id.fillna("").str.strip() != ""]
    return p.astype(str).reset_index(drop=True)


def _init_worker(dict_map, sk, skv):
    set_indic_resources(dict_map, sk, skv)


def _prep_chunk(args):
    ids, names, addrs, ctrys, src, want_vocab = args
    cols = {c: [] for c in PREP_COLS}
    voc = Counter()
    for eid, nm, ad, c in zip(ids, names, addrs, ctrys):
        if want_vocab:
            voc.update(stem(t) for t in name_tokens_v1(nm))
        toks = name_tokens(nm)
        st, w, n, u = parse_addr(ad, c)
        cols["entity_id"].append(eid); cols["src"].append(src); cols["country"].append(c)
        cols["state"].append(st or ""); cols["toks"].append(" ".join(toks))
        cols["words"].append(" ".join(w)); cols["nums"].append(" ".join(n)); cols["units"].append(" ".join(u))
        cols["has_addr"].append(bool((ad or "").strip()))
        cols["addr_ok"].append(bool((ad or "").strip()) and bool(w or n))
        cols["name_indic"].append(has_indic(nm)); cols["name_domain"].append(bool(DOMAIN_FLAG_RE.search(nm or "")))
        cols["legal"].append(legal_suffix_of(nm))
    return pd.DataFrame(cols), voc


def _run_chunks(df, src, n_workers, chunk, want_vocab):
    jobs = []
    for s in range(0, len(df), chunk):
        d = df.iloc[s:s + chunk]
        jobs.append((d.entity_id.tolist(), d.business_name.tolist(), d.business_address.tolist(),
                     d.country.tolist(), src, want_vocab))
    parts, voc = [], Counter()
    if n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        gc.collect(); gc.freeze()
        with ctx.Pool(n_workers, initializer=_init_worker,
                      initargs=(_IND["dict"], _IND["sk"], _IND["skv"])) as pool:
            for i, (p, v) in enumerate(pool.imap(_prep_chunk, jobs)):
                parts.append(p); voc.update(v)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    log(f"   {src}: {min((i + 1) * chunk, len(df)):,}/{len(df):,}")
        gc.unfreeze()
    else:
        for i, j in enumerate(jobs):
            p, v = _prep_chunk(j); parts.append(p); voc.update(v)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                log(f"   {src}: {min((i + 1) * chunk, len(df)):,}/{len(df):,}")
    return pd.concat(parts, ignore_index=True), voc


def prepare_split(split, data_dir, work_dir, indic_dict_path, parquet_dir=None,
                  n_workers=None, chunk=50_000, extra_vocab=None):
    """Clean + parse all three sources of a split. Cached as work_dir/prep/{split}_S{k}.parquet."""
    n_workers = n_workers or max(1, (os.cpu_count() or 2))
    pdir = os.path.join(work_dir, "prep"); os.makedirs(pdir, exist_ok=True)
    dict_map = json.load(open(indic_dict_path, encoding="utf-8")) if indic_dict_path and \
        os.path.exists(indic_dict_path) else {}
    log(f"[prepare {split}] Indic dictionary entries: {len(dict_map):,} | workers: {n_workers}")
    set_indic_resources(dict_map, {}, {})

    vpath = os.path.join(pdir, f"{split}_vocab.json")
    p1 = os.path.join(pdir, f"{split}_S1.parquet")
    if os.path.exists(p1) and os.path.exists(vpath):
        log(f"[prepare {split}] S1 cached"); voc = Counter(json.load(open(vpath)))
    else:
        raw = load_raw(split, 1, data_dir, parquet_dir); log(f"[prepare {split}] S1 raw {len(raw):,}")
        out, voc = _run_chunks(raw, "S1", n_workers, chunk, True)
        out.to_parquet(p1, index=False); json.dump(dict(voc), open(vpath, "w"))
        del raw, out; gc.collect()
    if extra_vocab:
        voc = voc + Counter(extra_vocab)
    sk, skv = build_vocab_index(voc)
    set_indic_resources(dict_map, sk, skv)
    for k in (2, 3):
        pk = os.path.join(pdir, f"{split}_S{k}.parquet")
        if os.path.exists(pk):
            log(f"[prepare {split}] S{k} cached"); continue
        raw = load_raw(split, k, data_dir, parquet_dir); log(f"[prepare {split}] S{k} raw {len(raw):,}")
        out, _ = _run_chunks(raw, f"S{k}", n_workers, chunk, False)
        out.to_parquet(pk, index=False)
        del raw, out; gc.collect()
    log(f"[prepare {split}] done")


def _prep_path(work_dir, split, k):
    return os.path.join(work_dir, "prep", f"{split}_S{k}.parquet")


def _read_cols(work_dir, split, k, cols):
    df = pd.read_parquet(_prep_path(work_dir, split, k), columns=cols)
    for c in cols:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            df[c] = df[c].fillna("").astype(str)
    return df


# =====================================================================================
# 6. FILLER DETECTOR (label-free) + PARTITIONS + SELECTION  (streamed, memory-light)
# =====================================================================================
def _doc_freq(series):
    c = Counter()
    for s in series:
        if s:
            c.update(set(s.split()))
    return c


def detect_fillers(split, work_dir, min_ratio=3.0, min_frac=0.00075, min_abs=30):
    """Words far more frequent in S2/S3 than in S1 (per country) -> injected fillers / junk.
    Reads one column at a time so memory stays low."""
    cnt = {}                                        # (country, field, is_s1) -> Counter
    n = Counter()                                   # (country, is_s1) -> records
    for k in (1, 2, 3):
        cty = _read_cols(work_dir, split, k, ["country"]).country.values
        for field in ("toks", "words"):
            col = _read_cols(work_dir, split, k, [field])[field].values
            for c in np.unique(cty):
                key = (c, field, k == 1)
                cnt.setdefault(key, Counter()).update(_doc_freq(col[cty == c]))
            del col; gc.collect()
        for c, v in zip(*np.unique(cty, return_counts=True)):
            n[(c, k == 1)] += int(v)
    out = {}
    for c in sorted({c for c, _ in n}):
        n1, n23 = max(n[(c, True)], 1), max(n[(c, False)], 1)
        res = {}
        for field in ("toks", "words"):
            c1, c23 = cnt.get((c, field, True), Counter()), cnt.get((c, field, False), Counter())
            thr = max(min_abs, min_frac * n23)
            res[field] = sorted(t for t, v in c23.items()
                                if v >= thr and (v / n23) / ((c1.get(t, 0) + 1) / n1) >= min_ratio)
        out[c] = res
        log(f"[fillers] {c}: name {len(res['toks'])} -> {res['toks'][:30]}")
        log(f"[fillers] {c}: addr {len(res['words'])} -> {res['words'][:30]}")
    return out


def apply_fillers(S, fillers):
    cty = S.country.astype(str).values
    for c, res in fillers.items():
        m = cty == c
        for field in ("toks", "words"):
            bad = set(res.get(field, []))
            if not bad or not m.any():
                continue
            vals = S[field].values[m]
            S.loc[m, field] = [" ".join(t for t in v.split() if t not in bad) if v else v for v in vals]
    return S


def assign_partitions(L, min_state_frac=0.5):
    """L needs src, country, state, addr_ok.
    country|state when the country's S1 mostly has a detectable state, else country|* ;
    S2/S3 without usable address (or without state in state-partitioned countries) -> country|POOL."""
    part = np.empty(len(L), dtype=object)
    src = L.src.astype(str).values; cty = L.country.astype(str).values
    st = L.state.astype(str).values; ok = L.addr_ok.values.astype(bool)
    for c in np.unique(cty):
        m = cty == c; is1 = m & (src == "S1"); m23 = m & (src != "S1")
        use_state = is1.sum() > 0 and (st[is1] != "").mean() >= min_state_frac
        if use_state:
            p = np.where(st != "", c + "|" + st.astype(object), c + "|*")
            p23 = np.where(ok & (st != ""), p, c + "|POOL")
        else:
            p = np.full(len(L), c + "|*", dtype=object)
            p23 = np.where(ok, p, c + "|POOL")
        part[is1] = p[is1]; part[m23] = p23[m23]
        log(f"[partitions] {c}: {'state partitions' if use_state else 'one block (no states detected)'}")
    return part


def choose_partitions(L, part, frac=1.0, seed=42, explicit=None):
    """Whole partitions per country until ~frac of its S1 is covered (all if frac>=1)."""
    s1 = part[(L.src.astype(str) == "S1").values]
    sizes = pd.Series(s1).value_counts()
    if explicit:
        return sorted(explicit)
    rng = np.random.RandomState(seed); chosen = set()
    for c in sorted({p.split("|")[0] for p in sizes.index}):
        ps = sorted(p for p in sizes.index if p.split("|")[0] == c)
        if frac >= 1:
            chosen |= set(ps); continue
        rng.shuffle(ps); tot, acc = sizes[ps].sum(), 0
        for p in ps:
            if acc >= frac * tot:
                break
            chosen.add(p); acc += sizes[p]
    return sorted(chosen)


def load_selection(split, work_dir, frac=1.0, seed=42, explicit=None, fillers=None):
    """Assign partitions from 4 light columns, choose partitions, then load only the selected rows."""
    light_cols = ["src", "country", "state", "addr_ok"]
    L = pd.concat([_read_cols(work_dir, split, k, light_cols) for k in (1, 2, 3)], ignore_index=True)
    part = assign_partitions(L)
    parts = choose_partitions(L, part, frac, seed, explicit)
    countries = {p.split("|")[0] for p in parts}
    keep = np.isin(part, parts) | np.isin(part, [c + "|POOL" for c in countries])
    sizes = [len(pd.read_parquet(_prep_path(work_dir, split, k), columns=["src"])) for k in (1, 2, 3)]
    offs = np.cumsum([0] + sizes)
    chunks = []
    for i, k in enumerate((1, 2, 3)):
        mk = keep[offs[i]:offs[i + 1]]
        df = pd.read_parquet(_prep_path(work_dir, split, k))
        df = df[mk].reset_index(drop=True)
        df["part"] = part[offs[i]:offs[i + 1]][mk]
        chunks.append(df); del df; gc.collect()
    S = pd.concat(chunks, ignore_index=True); del chunks, L; gc.collect()
    for c in ("toks", "words", "nums", "units", "state", "entity_id"):
        S[c] = S[c].fillna("").astype(str)
    for c in ("src", "country"):
        S[c] = S[c].astype(str).astype("category")
    S["ftoks"] = S["toks"].to_numpy(copy=True)             # full name tokens, kept for the model
    if fillers:
        S = apply_fillers(S, fillers)
    log(f"[select] partitions={len(parts)} | S1={int((S.src == 'S1').sum()):,} | "
        f"S2/S3={int((S.src != 'S1').sum()):,}")
    return S, parts


# =====================================================================================
# 6b. v3.0 RECORD-LEVEL RESOURCES (all label-free; computed per split on the FULL split)
# =====================================================================================
def _lseq_chunk(names):
    return [legal_set_of(n) for n in names]


def attach_name_flags(S, split, data_dir, work_dir, parquet_dir=None, n_workers=None, chunk=100_000):
    """S['lseq'] = canonical legal set of the RAW name (e.g. 'limited private'). Cached for the whole
    split in work_dir/prep/{split}_lseq.parquet (one raw read per source, once)."""
    fp = os.path.join(work_dir, "prep", f"{split}_lseq.parquet")
    if not os.path.exists(fp):
        n_workers = n_workers or max(1, os.cpu_count() or 1)
        out = []
        for k in (1, 2, 3):
            raw = load_raw(split, k, data_dir, parquet_dir)[["entity_id", "business_name"]]
            names = raw.business_name.tolist()
            jobs = [names[s:s + chunk] for s in range(0, len(names), chunk)]
            if n_workers > 1 and len(jobs) > 1:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(n_workers) as pool:
                    res = [x for part in pool.imap(_lseq_chunk, jobs) for x in part]
            else:
                res = [x for j in jobs for x in _lseq_chunk(j)]
            out.append(pd.DataFrame({"entity_id": raw.entity_id.values, "lseq": res}))
            log(f"[lseq {split}] S{k}: {len(raw):,}")
            del raw, names, jobs, res; gc.collect()
        pd.concat(out, ignore_index=True).to_parquet(fp, index=False); del out; gc.collect()
    d = pd.read_parquet(fp)
    S["lseq"] = pd.Series(d.lseq.values, index=d.entity_id.values).reindex(S.entity_id.values).fillna("").values
    del d; gc.collect()
    return S


def _name_keys(ctry, toks):
    return [f"{c}|{' '.join(sorted(set(t.split())))}" if t else None for c, t in zip(ctry, toks)]


def attach_name_counts(S, split, work_dir):
    """S['n1_same'] = number of S1 records (whole split, same country) whose name-token SET equals this
    record's; S['n23_same'] = same over S2/S3. Uses unfiltered name tokens (prep 'toks' == S.ftoks)."""
    fp = os.path.join(work_dir, "prep", f"{split}_namecnt.parquet")
    if not os.path.exists(fp):
        v1, v23 = None, None
        for k in (1, 2, 3):
            d = _read_cols(work_dir, split, k, ["country", "toks"])
            vc = pd.Series(_name_keys(d.country.values, d.toks.values)).value_counts()
            if k == 1:
                v1 = vc
            else:
                v23 = vc if v23 is None else v23.add(vc, fill_value=0)
            del d, vc; gc.collect()
        cnt = pd.DataFrame({"n1": v1, "n23": v23}).fillna(0)
        cnt.index.name = "key"
        cnt.reset_index().to_parquet(fp, index=False); del cnt, v1, v23; gc.collect()
    cnt = pd.read_parquet(fp).set_index("key")
    keys = pd.Series(_name_keys(S.country.astype(str).values, S.ftoks.values))
    S["n1_same"] = keys.map(cnt.n1).fillna(0).astype(np.float32).values
    S["n23_same"] = keys.map(cnt.n23).fillna(0).astype(np.float32).values
    del cnt, keys; gc.collect()
    return S


# ---- synthetic random names (Fayepyra, Solsyntavo, Irizephbelo, ...) ----
SYLL_SEED = """pyra lyra jax nexa nex wex zeph orbi kelo belo tavo drex flux halo brix onyx riza zeta yuma
delta vio veo mira faye quo iri cira avi lum syn umbra kor ecto nyla xylo evo sol arc vera calo""".split()
_SYLL = {"set": frozenset(SYLL_SEED), "common": frozenset()}


def _segment(t, inv, allow_unknown, minp=2, maxp=6):
    """Split t into inventory pieces; optionally ONE unknown piece (len 3..6).
    Returns (n_known, unknown_piece_or_None) preferring no unknown piece, or None."""
    n = len(t)

    @lru_cache(maxsize=None)
    def go(i, used):
        if i == n:
            return (0, None)
        best = None
        for L in range(minp, min(maxp, n - i) + 1):
            p = t[i:i + L]
            if p in inv:
                r = go(i + L, used)
                if r is not None:
                    c = (r[0] + 1, r[1])
                    if best is None or (c[1] is None, c[0]) > (best[1] is None, best[0]):
                        best = c
            if allow_unknown and not used and L >= 3:
                r = go(i + L, True)
                if r is not None:
                    c = (r[0], p)
                    if best is None or (c[1] is None, c[0]) > (best[1] is None, best[0]):
                        best = c
        return best
    return go(0, False)


def learn_syllables(split, work_dir, seed=SYLL_SEED, rounds=3, min_count=None):
    """Label-free: grow the syllable inventory from S2/S3 name tokens that are rare in S1 and are the
    seed syllables plus exactly one unknown piece (e.g. 'faye'+'pyra' teaches nothing new, 'quo'+'kelo'
    teaches 'quo' once it recurs in enough distinct tokens)."""
    voc = Counter(json.load(open(os.path.join(work_dir, "prep", f"{split}_vocab.json"))))
    cnt = Counter()
    for k in (2, 3):
        cnt.update(_doc_freq(_read_cols(work_dir, split, k, ["toks"]).toks.values))
    cands = [t for t in cnt if t.isascii() and t.isalpha() and 5 <= len(t) <= 24 and voc.get(t, 0) < 5]
    thr = min_count or max(25, int(2e-6 * sum(cnt.values())))
    inv = set(seed)
    for r in range(rounds):
        pieces = Counter()
        for t in cands:
            sg = _segment(t, inv, True)
            if sg is not None and sg[1] is not None and sg[0] >= 1:
                pieces[sg[1]] += 1
        new = {p for p, c in pieces.items() if c >= thr and voc.get(p, 0) < 50 and p not in inv}
        log(f"[syllables {split}] round {r + 1}: +{len(new)} {sorted(new)[:40]}")
        if not new:
            break
        inv |= new
    return sorted(inv)


def set_syllables(inv, common=None):
    _SYLL["set"] = frozenset(inv); _SYLL["common"] = frozenset(common or ())
    is_synth.cache_clear()


@lru_cache(maxsize=2_000_000)
def is_synth(t):
    if len(t) < 5 or not t.isascii() or not t.isalpha() or t in _SYLL["common"]:
        return False
    sg = _segment(t, _SYLL["set"], False)
    return sg is not None and sg[0] >= 2


# ---- glued names split against the S1 vocabulary (smarthayesxcel -> smart hayes xcel) ----
_SEG = {"voc": {}}


def set_split_vocab(voc):
    _SEG["voc"] = {t: c for t, c in voc.items() if c >= 3 and len(t) >= 3 and t.isascii() and t.isalpha()}
    split_token.cache_clear()


@lru_cache(maxsize=2_000_000)
def split_token(t):
    V = _SEG["voc"]
    if len(t) < 7 or t in V or not t.isascii() or not t.isalpha():
        return (t,)
    n = len(t); best = [None] * (n + 1); best[0] = (0.0, ())
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 3, min(n, i + 20) + 1):
            w = t[i:j]
            c = V.get(w) or (V.get(stem(w)) if j == n else None)
            if c:
                sc = best[i][0] + math.log(c) - 3.0
                if best[j] is None or sc > best[j][0]:
                    best[j] = (sc, best[i][1] + (w if w in V else stem(w),))
    if best[n] is None or len(best[n][1]) < 2:
        return (t,)
    return best[n][1]


def setup_v3_resources(S, split, data_dir, work_dir, parquet_dir=None, extra_syllables=None):
    """Everything FEATURES_V3 needs, attached to S / set as module globals (set BEFORE forking workers).
    Returns the syllable inventory used."""
    t0 = time.time()
    S = attach_name_flags(S, split, data_dir, work_dir, parquet_dir)
    S = attach_name_counts(S, split, work_dir)
    voc = Counter(json.load(open(os.path.join(work_dir, "prep", f"{split}_vocab.json"))))
    sp = os.path.join(work_dir, f"syllables_{split}.json")
    if os.path.exists(sp):
        inv = json.load(open(sp))
    else:
        inv = learn_syllables(split, work_dir); json.dump(inv, open(sp, "w"))
    inv = sorted(set(inv) | set(extra_syllables or []))
    set_syllables(inv, common={t for t, c in voc.items() if c >= 20})
    set_split_vocab(voc)
    log(f"[v3 resources {split}] syllables={len(inv)} | split vocab={len(_SEG['voc']):,} | "
        f"{time.time() - t0:.0f}s")
    return S, inv



# =====================================================================================
# 7. BLOCKING
# =====================================================================================
FAMS = ["N_set", "N_pair", "N_glue6", "A_numword", "A_word2"]


def del1(n):
    out = {n}
    if len(n) >= 2:
        for i in range(len(n)):
            v = n[:i] + n[i + 1:]
            if v:
                out.add(str(int(v)))
    return out


def _rec_keys(toks, words, nums, dfn, dfa, is_s1, with_addr):
    out = []
    if toks:
        out.append((0, " ".join(sorted(set(toks)))))
        gl = fold("".join(toks))
        if len(gl) >= 4:
            out.append((2, gl[:6]))
    r = sorted([t for t in set(toks) if is_s1 or t in dfn], key=lambda t: (dfn.get(t, 0), t))
    top = r[:4]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = sorted((top[i], top[j])); out.append((1, a + "|" + b))
    if with_addr:
        rw = sorted([w for w in set(words) if is_s1 or w in dfa], key=lambda w: (dfa.get(w, 0), w))[:2 if is_s1 else 3]
        for n in nums[:3]:
            for v in (del1(n) if len(n) >= 3 else (n,)):
                for w in rw:
                    out.append((3, v + "|" + w))
        for i in range(len(rw)):
            for j in range(i + 1, len(rw)):
                a, b = sorted((rw[i], rw[j])); out.append((4, a + "|" + b))
    return out


def _key_table(pos, toks, words, nums, gkey, dfn, dfa, is_s1, with_addr):
    I, F, K = [], [], []
    for p in pos:
        for f, k in _rec_keys(toks[p], words[p], nums[p], dfn, dfa, is_s1, with_addr):
            I.append(p); F.append(f); K.append(hash((gkey, f, k)))
    return pd.DataFrame({"i": np.array(I, np.int64), "fam": np.array(F, np.int8), "key": np.array(K, np.int64)})


def _key_pairs(p1, p23, toks, words, nums, gkey, with_addr, cap):
    dfn = Counter(t for p in p1 for t in set(toks[p]))
    dfa = Counter(w for p in p1 for w in set(words[p])) if with_addr else {}
    k1 = _key_table(p1, toks, words, nums, gkey, dfn, dfa, True, with_addr)
    k2 = _key_table(p23, toks, words, nums, gkey, dfn, dfa, False, with_addr)
    bs = k2.groupby("key").size()
    k2 = k2[k2.key.isin(bs.index[bs.values <= cap])]
    m = k1.merge(k2, on=["key", "fam"], suffixes=("1", "2"))[["i1", "i2", "fam"]].drop_duplicates()
    if len(m) == 0:
        return pd.DataFrame({"i1": [], "i2": [], "bits": []}).astype({"i1": np.int64, "i2": np.int64, "bits": np.int16})
    m["bits"] = np.left_shift(np.int16(1), m.fam.values.astype(np.int16)).astype(np.int16)
    return m.groupby(["i1", "i2"], sort=False).bits.sum().reset_index()


def _tfidf(docs, **kw):
    from sklearn.feature_extraction.text import TfidfVectorizer
    for md in (2, 1):
        try:
            return TfidfVectorizer(min_df=md, dtype=np.float32, sublinear_tf=True, **kw).fit_transform(docs).tocsr()
        except ValueError:
            continue
    from scipy.sparse import csr_matrix
    return csr_matrix((len(docs), 1), dtype=np.float32)


def _rowdot(X, a, b, chunk=2_000_000):
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.asarray(X[a[s:s + chunk]].multiply(X[b[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def _topn(A, B, n, thr=0.05, chunk=20_000, nth=None):
    import sparse_dot_topn as sdt
    if A.shape[0] == 0 or B.shape[0] == 0 or A.nnz == 0 or B.nnz == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    BT = B.T.tocsr(); rows, cols = [], []
    nth = nth or os.cpu_count() or 1
    for s in range(0, A.shape[0], chunk):
        Rm = sdt.sp_matmul_topn(A[s:s + chunk], BT, top_n=n, threshold=thr, sort=True, n_threads=nth).tocoo()
        rows.append(Rm.row.astype(np.int64) + s); cols.append(Rm.col.astype(np.int64))
    return np.concatenate(rows), np.concatenate(cols)


def _ranks_within(g, score):
    order = np.lexsort((-score, g)); gs = g[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    rank = np.empty(len(g), np.int32)
    rank[order] = np.arange(len(gs)) - np.repeat(starts, np.diff(np.r_[starts, len(gs)]))
    return rank


def _block_job(pos1, pos23, toks, words, nums, gkey, with_addr, cfg):
    """Keys + kNN for one job; returns candidates with global row indices and scores."""
    if len(pos1) == 0 or len(pos23) == 0:
        return None
    Kp = _key_pairs(pos1, pos23, toks, words, nums, gkey, with_addr, cfg["cap"])
    allpos = np.concatenate([pos1, pos23])
    loc = np.full(int(allpos.max()) + 1, -1, np.int64); loc[allpos] = np.arange(len(allpos))
    ndocs = [" ".join(toks[p]) for p in allpos]
    Xn = _tfidf(ndocs, analyzer="char_wb", ngram_range=(3, 3))
    adocs = [" ".join(words[p] + nums[p]) for p in allpos]
    Xa = _tfidf(adocs, tokenizer=str.split, token_pattern=None, lowercase=False)
    n1 = len(pos1)
    if cfg["use_knn"]:
        if with_addr:
            from scipy.sparse import hstack
            Xc = (hstack([Xn, Xa]).tocsr() * np.float32(1 / np.sqrt(2)))
            r, c = _topn(Xc[:n1], Xc[n1:], cfg["knn_state"], nth=cfg.get("threads"))
        else:
            r, c = _topn(Xn[:n1], Xn[n1:], cfg["knn_noaddr"], nth=cfg.get("threads"))
        Kn = pd.DataFrame({"i1": pos1[r], "i2": pos23[c], "knn": True})
    else:
        Kn = pd.DataFrame({"i1": np.zeros(0, np.int64), "i2": np.zeros(0, np.int64), "knn": np.zeros(0, bool)})
    C = Kp.merge(Kn, on=["i1", "i2"], how="outer")
    C["bits"] = C.bits.fillna(0).astype(np.int16)
    C["knn"] = C.knn.eq(True).astype(bool)
    if len(C) == 0:
        return None
    la = loc[C.i1.values.astype(np.int64)]
    lb = loc[C.i2.values.astype(np.int64)]
    C["sn"] = _rowdot(Xn, la, lb); C["sa"] = _rowdot(Xa, la, lb)
    score = (C.sn + C.sa).values if with_addr else C.sn.values
    C["r0"] = _ranks_within(C.i1.values, score)
    K = cfg["k_state"] if with_addr else cfg["k_noaddr"]
    C = C[C.r0 < K].drop(columns="r0")
    C["pool"] = not with_addr
    return C.reset_index(drop=True)


BLOCK_DEFAULTS = dict(cap=200, use_knn=True, knn_state=20, knn_noaddr=10, k_state=30, k_noaddr=5)
_EMPTY_C = {"i1": np.int64, "i2": np.int64, "bits": np.int16, "knn": bool, "sn": np.float32, "sa": np.float32,
            "pool": bool}


def _block_worker(args):
    """One blocking job in its own process: gets only its own records (as strings), saves its candidates."""
    pk, with_addr, pos1, pos23, t_str, w_str, n_str, cfg, fn = args
    t0 = time.time()
    toks = [s.split() if s else [] for s in t_str]
    words = [s.split() if s else [] for s in w_str]
    nums = [s.split() if s else [] for s in n_str]
    n1 = len(pos1)
    C = _block_job(np.arange(n1), np.arange(n1, n1 + len(pos23)), toks, words, nums, pk, with_addr, cfg)
    if C is None:
        C = pd.DataFrame({k: np.zeros(0, t) for k, t in _EMPTY_C.items()})
    else:
        allpos = np.concatenate([pos1, pos23]).astype(np.int64)
        C["i1"] = allpos[C.i1.values.astype(np.int64)]; C["i2"] = allpos[C.i2.values.astype(np.int64)]
    C.to_parquet(fn, index=False)
    return pk, n1, len(pos23), len(C), time.time() - t0


def run_blocking(S, parts, run_dir, cfg=None, n_workers=1):
    """Blocking per partition (+ no-address pool per country). Each job is saved as soon as it finishes
    (resumable). n_workers>1 runs several partitions at once, largest first."""
    cfg = {**BLOCK_DEFAULTS, **(cfg or {})}
    ncpu = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))
    cfg.setdefault("threads", max(1, ncpu // n_workers))
    bdir = os.path.join(run_dir, "blocks"); os.makedirs(bdir, exist_ok=True)
    is1 = (S.src == "S1").values
    part = S.part.values
    cty = S.country.astype(str).values
    T, W, N = S.toks.values, S.words.values, S.nums.values
    jobs = [(p, True) for p in parts] + [(c + "|POOL", False) for c in sorted({p.split("|")[0] for p in parts})]
    todo = []
    for pk, with_addr in jobs:
        fn = os.path.join(bdir, re.sub(r"[^A-Za-z0-9_|*-]", "_", pk).replace("|", "__").replace("*", "ALL") + ".parquet")
        if os.path.exists(fn):
            continue
        if with_addr:
            pos1 = np.flatnonzero(is1 & (part == pk)); pos23 = np.flatnonzero(~is1 & (part == pk))
        else:
            pos1 = np.flatnonzero(is1 & (cty == pk.split("|")[0])); pos23 = np.flatnonzero(~is1 & (part == pk))
        todo.append((pk, with_addr, pos1, pos23, fn))
    todo.sort(key=lambda j: -(len(j[2]) * max(len(j[3]), 1)))
    log(f"[block] {len(jobs) - len(todo)} jobs cached, {len(todo)} to run | workers={n_workers} "
        f"| kNN threads/job={cfg['threads']}")

    def args():
        for pk, wa, p1, p23, fn in todo:
            allpos = np.concatenate([p1, p23])
            yield (pk, wa, p1, p23, T[allpos].tolist(), W[allpos].tolist(), N[allpos].tolist(), cfg, fn)

    def report(r):
        pk, a, b, n, dt = r
        log(f"[block] {pk:28s} S1={a:>9,} S2/S3={b:>9,} -> {n:>11,} cand ({dt:.0f}s)")

    if n_workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        gc.collect(); gc.freeze()            # keep forked workers from copying the parent's memory
        try:
            with mp.get_context("fork").Pool(n_workers, maxtasksperchild=1) as pool:
                for r in pool.imap_unordered(_block_worker, args()):
                    report(r)
        finally:
            gc.unfreeze()
    else:
        for a in args():
            report(_block_worker(a))
    C = pd.concat([pd.read_parquet(os.path.join(bdir, f)) for f in sorted(os.listdir(bdir))], ignore_index=True)
    C = C.sort_values(["i1", "i2", "pool"]).drop_duplicates(["i1", "i2"]).reset_index(drop=True)
    return C


# =====================================================================================
# 8. FEATURES
# =====================================================================================
# stage-1 base features (unchanged since v1.0, so saved feature files stay valid)
FEATURES = ["sn", "sa", "ssum", "n_a", "n_b", "tok_inter", "jacc", "cont_a", "cont_b", "set_equal",
            "idf_ov", "idf_max_shared", "idf_b_unshared", "b_oov_frac",
            "glue_equal", "glue_contain", "glue_prefix", "tsr", "g_ratio", "g_partial", "g_jw",
            "a_has_addr", "b_has_addr", "w_inter", "w_jacc", "w_idf_ov", "a_nnum", "b_nnum",
            "num_exact", "num_del1", "num_first_eq", "num_conflict", "unit_both", "unit_share", "unit_conflict",
            "b_s3", "b_indic", "b_domain", "f_set", "f_pair", "f_glue", "f_numword", "f_word2", "knn", "pool",
            "r1", "gap1", "n_c1", "r2", "gap2", "n_c2", "mutual"]
# v2 fix 1 + 2: full names (detected filler words kept) and strict house-number comparison
# v2.1: best-pair token fuzzy distance (min_char_dist) - catches single-word typos (Nevada/Nevbada,
# Republic/Repulbic) that get diluted inside whole-string / set-based scores. Appended at the end so
# existing feature positions/names for v2.0-trained models and cached feature files stay unaffected.
FEATURES_EXTRA = ["full_jacc", "full_equal", "extra_b_full", "extra_a_full", "b_filler_extra",
                  "num_seq_equal", "num_first_absdiff", "num_first_trunc", "num_first_diffpos",
                  "num_b_subset_a", "num_first_lendiff", "min_char_dist",
                  # v2.2: legal-suffix family features - catches the "name matches, no address to
                  # confirm identity, legal suffix swapped to a DIFFERENT family" false-merge pattern
                  # (Ltd->Corp, PC->Center Inc, Limited->LLP) seen throughout the v2.1 error log, while
                  # NOT penalising the routine within-family swaps the EDA found in true matches
                  # (pc->inc, corp->corporation, limited->ltd - see _LEGAL_FAMILY).
                  "legal_a_present", "legal_b_present", "legal_changed", "legal_changed_no_addr",
                  "legal_asym_no_addr"]
# v3.0 block (cached separately in run_dir/features_v3; v2.3's emb_cos removed)
FEATURES_V3 = ["b_synth_frac", "b_synth_all", "a_synth_frac", "split_jacc", "split_equal", "split_gain",
               "num_set_jacc", "num_b_unmatched", "addr_exact", "addr_words_equal",
               "priv_a", "priv_b", "priv_conflict", "pub_conflict", "lseq_diff", "lseq_diff_no_addr",
               "a_name_n1", "b_name_n1", "b_name_n23", "f_expand", "exp_sib", "exp_kaddr"]
FEATURES_S1 = FEATURES + FEATURES_EXTRA + FEATURES_V3
# v2 fix 3: second pass, using the S1's other confident candidates (siblings) and model competition
# v2.1: sib_addr_conflict - is this candidate's address a near-exact match to a DIFFERENT S1's top
# pick (the same-building-different-business false-merge pattern), appended at the end for the same
# backward-compatibility reason as above.
SIB_FEATURES = ["p1", "p1_r1", "p1_gap1", "p1_r2", "p1_gap2", "p1_other_best", "sib_n", "sib_maxp",
                "sib_name_jacc", "sib_glue_eq", "sib_addr_jacc", "sib_num_eq", "sib_addr_conflict",
                "sib_tsr_max", "sib_addr_exact", "b_conf_elsewhere"]
FEATURES_S2 = FEATURES_S1 + SIB_FEATURES
MONO = {"sn": 1, "sa": 1, "ssum": 1, "jacc": 1, "idf_ov": 1, "set_equal": 1, "glue_equal": 1, "tsr": 1,
        "w_jacc": 1, "w_idf_ov": 1, "num_exact": 1, "num_conflict": -1, "unit_conflict": -1,
        "full_jacc": 1, "num_seq_equal": 1, "b_filler_extra": -1, "p1": 1, "sib_name_jacc": 1,
        "min_char_dist": -1, "sib_addr_conflict": -1,
        "legal_changed": -1, "legal_changed_no_addr": -1, "legal_asym_no_addr": -1}


def build_resources(S, fillers=None):
    """Label-free statistics from S1 (per country): document frequencies + detected filler words."""
    res = {}
    for ctry in S.country.cat.categories:
        s1 = S[(S.country == ctry) & (S.src == "S1")]
        res[ctry] = {"n": max(len(s1), 1), "dfn": _doc_freq(s1.toks), "dfa": _doc_freq(s1.words),
                     "fill": set((fillers or {}).get(ctry, {}).get("toks", []))}
    return res


def attach_full_tokens(S, split, work_dir):
    """Adds ftoks = name tokens BEFORE filler removal (for selections saved by v1.0)."""
    if "ftoks" in S.columns:
        return S
    ids, toks = [], []
    for k in (1, 2, 3):
        d = pd.read_parquet(_prep_path(work_dir, split, k), columns=["entity_id", "toks"])
        m = d.entity_id.isin(S.entity_id.values)
        ids.append(d.entity_id.values[m]); toks.append(d.toks.fillna("").values[m])
        del d; gc.collect()
    mp_ = pd.Series(np.concatenate(toks), index=np.concatenate(ids))
    S["ftoks"] = mp_.reindex(S.entity_id.values).fillna("").astype(str).values
    return S


def _pair_loop(at, bt, aw, bw, an, bn, au, bu, ctry, res):
    n = len(at)
    F = {k: np.zeros(n, np.float32) for k in
         ["n_a", "n_b", "tok_inter", "jacc", "cont_a", "cont_b", "set_equal", "idf_ov", "idf_max_shared",
          "idf_b_unshared", "b_oov_frac", "glue_equal", "glue_contain", "glue_prefix", "w_inter", "w_jacc",
          "w_idf_ov", "a_nnum", "b_nnum", "num_exact", "num_del1", "num_first_eq", "num_conflict",
          "unit_both", "unit_share", "unit_conflict"]}
    ga, gb = [""] * n, [""] * n
    cache_idf = {}
    for k in range(n):
        c = ctry[k]
        if c not in cache_idf:
            r = res[c]; N = r["n"]
            cache_idf[c] = (r["dfn"], r["dfa"], N, math.log(N + 1))
        dfn, dfa, N, maxidf = cache_idf[c]
        A, B = at[k].split(), bt[k].split()
        sA, sB = set(A), set(B)
        inter, uni = sA & sB, sA | sB
        F["n_a"][k] = len(sA); F["n_b"][k] = len(sB); F["tok_inter"][k] = len(inter)
        if uni:
            F["jacc"][k] = len(inter) / len(uni)
            F["set_equal"][k] = float(sA == sB)
            idf = {t: (math.log((N + 1) / (dfn.get(t, 0) + 1))) for t in uni}
            su = sum(idf.values())
            si = sum(idf[t] for t in inter)
            F["idf_ov"][k] = si / su if su > 0 else 0
            F["idf_max_shared"][k] = max((idf[t] for t in inter), default=0.0)
            F["idf_b_unshared"][k] = sum(idf[t] for t in sB - sA)
        if sA:
            F["cont_a"][k] = len(inter) / len(sA)
        if sB:
            F["cont_b"][k] = len(inter) / len(sB)
            F["b_oov_frac"][k] = sum(1 for t in sB if t not in dfn) / len(sB)
        g1, g2 = fold("".join(A)), fold("".join(B))
        ga[k], gb[k] = g1, g2
        if g1 and g2:
            F["glue_equal"][k] = float(g1 == g2)
            F["glue_contain"][k] = float(min(len(g1), len(g2)) >= 4 and (g1 in g2 or g2 in g1))
            m = 0
            for x, y in zip(g1, g2):
                if x != y:
                    break
                m += 1
            F["glue_prefix"][k] = m / min(len(g1), len(g2))
        WA, WB = set(aw[k].split()), set(bw[k].split())
        wi, wu = WA & WB, WA | WB
        F["w_inter"][k] = len(wi)
        if wu:
            F["w_jacc"][k] = len(wi) / len(wu)
            idw = {t: math.log((N + 1) / (dfa.get(t, 0) + 1)) for t in wu}
            su = sum(idw.values())
            F["w_idf_ov"][k] = sum(idw[t] for t in wi) / su if su > 0 else 0
        NA, NB = an[k].split(), bn[k].split()
        F["a_nnum"][k] = len(NA); F["b_nnum"][k] = len(NB)
        if NA and NB:
            ex = bool(set(NA) & set(NB))
            if ex:
                F["num_exact"][k] = 1; d1 = True
            else:
                da = set().union(*[del1(x) for x in NA]); db = set().union(*[del1(x) for x in NB])
                d1 = bool(da & db)
                F["num_del1"][k] = float(d1)
            F["num_first_eq"][k] = float(NA[0] == NB[0])
            F["num_conflict"][k] = float(not ex and not d1)
        UA, UB = set(au[k].split()), set(bu[k].split())
        if UA and UB:
            F["unit_both"][k] = 1
            sh = bool(UA & UB)
            F["unit_share"][k] = float(sh); F["unit_conflict"][k] = float(not sh)
    return F, ga, gb



def _num_int(x):
    try:
        return int(x)
    except ValueError:
        return None


def _min_char_dist(ea, eb):
    """Best (lowest) normalised Levenshtein distance among the leftover token pairs after exact-token
    overlap is removed (ea = S1-only tokens, eb = S2/S3-only tokens). Catches single-word typos
    (Nevada/Nevbada, Republic/Repulbic) that get diluted inside whole-string / set-based scores.
    0.0 when there is nothing left to compare (all tokens already matched on at least one side) -
    matches the 'no mismatch signal' meaning, not an unknown/undefined value."""
    if not ea or not eb:
        return 0.0
    from rapidfuzz.distance import Levenshtein
    best = 1.0
    for x in ea:
        for y in eb:
            d = Levenshtein.normalized_distance(x, y, score_cutoff=best)
            if d < best:
                best = d
    return best


def _pair_loop_extra(fa, fb, an, bn, ctry, res, la=None, lb=None, b_has_addr=None):
    n = len(fa)
    F = {k: np.zeros(n, np.float32) for k in FEATURES_EXTRA}
    F["num_first_absdiff"][:] = -1; F["num_first_diffpos"][:] = -1; F["num_first_lendiff"][:] = -1
    for k in range(n):
        A, B = set(fa[k].split()), set(fb[k].split())
        u = A | B
        if u:
            F["full_jacc"][k] = len(A & B) / len(u)
            F["full_equal"][k] = float(A == B)
        eb, ea = B - A, A - B
        F["extra_b_full"][k] = len(eb); F["extra_a_full"][k] = len(ea)
        F["min_char_dist"][k] = _min_char_dist(ea, eb)
        fill = res[ctry[k]]["fill"]
        if fill:
            F["b_filler_extra"][k] = sum(1 for t in eb if t in fill)
        NA, NB = an[k].split(), bn[k].split()
        if NA and NB:
            F["num_seq_equal"][k] = float(NA == NB)
            F["num_b_subset_a"][k] = float(set(NB) <= set(NA))
            x, y = NA[0], NB[0]
            xi, yi = _num_int(x), _num_int(y)
            if xi is not None and yi is not None:
                F["num_first_absdiff"][k] = math.log1p(abs(xi - yi))
            F["num_first_trunc"][k] = float(x != y and (x.startswith(y) or y.startswith(x)
                                                        or x.endswith(y) or y.endswith(x)))
            F["num_first_lendiff"][k] = abs(len(x) - len(y))
            if len(x) == len(y):
                F["num_first_diffpos"][k] = sum(1 for c1, c2 in zip(x, y) if c1 != c2)
        if la is not None and lb is not None:
            fam_a, fam_b = la[k], lb[k]
            F["legal_a_present"][k] = float(bool(fam_a))
            F["legal_b_present"][k] = float(bool(fam_b))
            changed = bool(fam_a) and bool(fam_b) and fam_a != fam_b
            F["legal_changed"][k] = float(changed)
            no_addr = b_has_addr is not None and not b_has_addr[k]
            if changed and no_addr:
                F["legal_changed_no_addr"][k] = 1.0
            elif no_addr and bool(fam_a) != bool(fam_b):
                F["legal_asym_no_addr"][k] = 1.0
    return F


def _pair_loop_v3(fa, fb, wa, wb, na, nb, la, lb, b_has_addr, n1a, n1b, n23b, bits, esib, ekad):
    n = len(fa)
    F = {k: np.zeros(n, np.float32) for k in FEATURES_V3}
    F["num_set_jacc"][:] = -1; F["num_b_unmatched"][:] = -1
    for k in range(n):
        A, B = fa[k].split(), fb[k].split()
        if A:
            F["a_synth_frac"][k] = sum(map(is_synth, A)) / len(A)
        if B:
            sb = sum(map(is_synth, B))
            F["b_synth_frac"][k] = sb / len(B); F["b_synth_all"][k] = float(sb == len(B))
        sA, sB = set(A), set(B)
        xA, xB = set(), set()
        for t in sA:
            xA.update(split_token(t))
        for t in sB:
            xB.update(split_token(t))
        u = xA | xB
        if u:
            sj = len(xA & xB) / len(u)
            F["split_jacc"][k] = sj; F["split_equal"][k] = float(xA == xB)
            uo = sA | sB
            F["split_gain"][k] = sj - (len(sA & sB) / len(uo) if uo else 0.0)
        NA, NB = set(na[k].split()), set(nb[k].split())
        if NA and NB:
            F["num_set_jacc"][k] = len(NA & NB) / len(NA | NB)
            F["num_b_unmatched"][k] = len(NB - NA)
        WA, WB = set(wa[k].split()), set(wb[k].split())
        weq = bool(WA) and WA == WB
        F["addr_words_equal"][k] = float(weq)
        F["addr_exact"][k] = float(weq and bool(NA) and NA == NB)
        LA, LB = set(la[k].split()), set(lb[k].split())
        pa, pb = "private" in LA, "private" in LB
        ha, hb = bool(LA - {"public"}), bool(LB - {"public"})
        F["priv_a"][k] = float(pa); F["priv_b"][k] = float(pb)
        both = ha and hb
        F["priv_conflict"][k] = float(both and pa != pb)
        F["pub_conflict"][k] = float(both and (("public" in LA) != ("public" in LB)))
        d = both and LA != LB
        F["lseq_diff"][k] = float(d)
        F["lseq_diff_no_addr"][k] = float(d and not b_has_addr[k])
    F["a_name_n1"] = np.log1p(n1a).astype(np.float32)
    F["b_name_n1"] = np.log1p(n1b).astype(np.float32)
    F["b_name_n23"] = np.log1p(n23b).astype(np.float32)
    F["f_expand"] = ((bits >> 5) & 1).astype(np.float32)
    F["exp_sib"] = esib.astype(np.float32); F["exp_kaddr"] = ekad.astype(np.float32)
    return F


_FX = {}          # shared (fork-inherited, read-only) state for parallel feature chunks


def _arrow(values):
    import pyarrow as pa
    return pa.array([v if isinstance(v, str) else "" for v in values], type=pa.large_string())


def _take(arr, idx):
    import pyarrow as pa
    return arr.take(pa.array(idx, type=pa.int64())).to_pylist()


def _setup_features(C, S, res):
    """Per-record data (Arrow strings: no copy-on-write growth in forked workers) + competition features.
    v3.0: also the FEATURES_V3 inputs (S.lseq / n1_same / n23_same from setup_v3_resources; C.exp_sib /
    C.exp_kaddr from expand_candidates - zeros when absent)."""
    s = (C.sn.values + C.sa.values).astype(np.float32)
    zS, zC = np.zeros(len(S), np.float32), np.zeros(len(C), np.float32)
    comp = pd.DataFrame({"i1": C.i1.values, "i2": C.i2.values, "s": s})
    g1, g2 = comp.groupby("i1").s, comp.groupby("i2").s
    cats = list(S.country.cat.categories)
    _FX.clear()
    _FX.update(
        toks=_arrow(S.toks.values), ftoks=_arrow(S.ftoks.values if "ftoks" in S.columns else S.toks.values),
        words=_arrow(S.words.values), nums=_arrow(S.nums.values), units=_arrow(S.units.values),
        legal=_arrow(S.legal.values if "legal" in S.columns else np.full(len(S), "", dtype=object)),
        ccode=S.country.cat.codes.values.astype(np.int16), cats=cats,
        has_addr=(S.has_addr.values & ((S.words.values != "") | (S.nums.values != ""))).astype(np.float32),
        b_s3=(S.src == "S3").values.astype(np.float32),
        b_indic=S.name_indic.values.astype(np.float32), b_dom=S.name_domain.values.astype(np.float32),
        i1=C.i1.values.astype(np.int64), i2=C.i2.values.astype(np.int64),
        sn=C.sn.values.astype(np.float32), sa=C.sa.values.astype(np.float32),
        bits=C.bits.values.astype(np.int16), knn=C.knn.values.astype(np.float32),
        pool=C.pool.values.astype(np.float32),
        r1=(g1.rank(method="first", ascending=False).values - 1).astype(np.float32),
        gap1=(g1.transform("max").values - s).astype(np.float32),
        n1=g1.transform("size").values.astype(np.float32),
        r2=(g2.rank(method="first", ascending=False).values - 1).astype(np.float32),
        gap2=(g2.transform("max").values - s).astype(np.float32),
        n2=g2.transform("size").values.astype(np.float32),
        lseq=_arrow(S.lseq.values if "lseq" in S.columns else np.full(len(S), "", dtype=object)),
        n1s=S.n1_same.values.astype(np.float32) if "n1_same" in S.columns else zS,
        n23s=S.n23_same.values.astype(np.float32) if "n23_same" in S.columns else zS,
        esib=C.exp_sib.values.astype(np.float32) if "exp_sib" in C.columns else zC,
        ekad=C.exp_kaddr.values.astype(np.float32) if "exp_kaddr" in C.columns else zC,
        res=res)
    del comp, g1, g2; gc.collect()


def _feature_chunk(job):
    """job = (key, candidate-row indices, rapidfuzz workers, mode) -> float32 matrix.
    mode 'base' = FEATURES, 'extra' = FEATURES_EXTRA, 'all' = FEATURES_S1."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    key, rows, workers, mode = job
    X = _FX
    a = X["i1"][rows]; b = X["i2"][rows]
    cats = X["cats"]; ctry = [cats[c] for c in X["ccode"][a]]
    out = {}
    if mode in ("base", "all"):
        ta, tb = _take(X["toks"], a), _take(X["toks"], b)
        wa, wb = _take(X["words"], a), _take(X["words"], b)
        na, nb = _take(X["nums"], a), _take(X["nums"], b)
        ua, ub = _take(X["units"], a), _take(X["units"], b)
        F, ga, gb = _pair_loop(ta, tb, wa, wb, na, nb, ua, ub, ctry, X["res"])
        F["sn"] = X["sn"][rows]; F["sa"] = X["sa"][rows]; F["ssum"] = F["sn"] + F["sa"]
        F["tsr"] = process.cpdist(ta, tb, scorer=fuzz.token_set_ratio, workers=workers).astype(np.float32)
        F["g_ratio"] = process.cpdist(ga, gb, scorer=fuzz.ratio, workers=workers).astype(np.float32)
        F["g_partial"] = process.cpdist(ga, gb, scorer=fuzz.partial_ratio, workers=workers).astype(np.float32)
        F["g_jw"] = process.cpdist(ga, gb, scorer=JaroWinkler.normalized_similarity, workers=workers).astype(np.float32)
        F["a_has_addr"] = X["has_addr"][a]; F["b_has_addr"] = X["has_addr"][b]
        F["b_s3"] = X["b_s3"][b]; F["b_indic"] = X["b_indic"][b]; F["b_domain"] = X["b_dom"][b]
        bits = X["bits"][rows]
        for i, nm in enumerate(["f_set", "f_pair", "f_glue", "f_numword", "f_word2"]):
            F[nm] = ((bits >> i) & 1).astype(np.float32)
        F["knn"] = X["knn"][rows]; F["pool"] = X["pool"][rows]
        F["r1"] = X["r1"][rows]; F["gap1"] = X["gap1"][rows]; F["n_c1"] = X["n1"][rows]
        F["r2"] = X["r2"][rows]; F["gap2"] = X["gap2"][rows]; F["n_c2"] = X["n2"][rows]
        F["mutual"] = ((F["r1"] == 0) & (F["r2"] == 0)).astype(np.float32)
        out.update(F)
    else:
        na, nb = _take(X["nums"], a), _take(X["nums"], b)
    if mode in ("extra", "all"):
        la, lb = _take(X["legal"], a), _take(X["legal"], b)
        b_has_addr = X["has_addr"][b]
        out.update(_pair_loop_extra(_take(X["ftoks"], a), _take(X["ftoks"], b), na, nb, ctry, X["res"],
                                    la, lb, b_has_addr))
    if mode in ("v3", "all"):
        out.update(_pair_loop_v3(_take(X["ftoks"], a), _take(X["ftoks"], b),
                                 _take(X["words"], a), _take(X["words"], b), na, nb,
                                 _take(X["lseq"], a), _take(X["lseq"], b), X["has_addr"][b],
                                 X["n1s"][a], X["n1s"][b], X["n23s"][b], X["bits"][rows],
                                 X["esib"][rows], X["ekad"][rows]))
    cols = {"base": FEATURES, "extra": FEATURES_EXTRA, "v3": FEATURES_V3, "all": FEATURES_S1}[mode]
    return key, np.column_stack([np.asarray(out[k], np.float32) for k in cols])


def _chunk_iter(jobs, n_workers):
    """jobs: list of (key, rows, mode). Yields (key, matrix); parallel via fork when n_workers > 1."""
    if n_workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        gc.collect(); gc.freeze()
        try:
            with mp.get_context("fork").Pool(n_workers) as pool:
                for r in pool.imap_unordered(_feature_chunk, [(k, rows, 1, m) for k, rows, m in jobs]):
                    yield r
        finally:
            gc.unfreeze()
        return
    for k, rows, m in jobs:
        yield _feature_chunk((k, rows, -1, m))


def _default_workers():
    return max(1, min(12, (os.cpu_count() or 2) // 2))


def build_features(C, S, res, run_dir, chunk=500_000, n_workers=None, mode="base"):
    """Feature DataFrame aligned with C; parts cached on disk (resumable).
    mode 'base' -> run_dir/features (52 v1 features), 'extra' -> run_dir/features_extra (v2 additions),
    'v3' -> run_dir/features_v3 (v3.0 block; call setup_v3_resources first)."""
    sub = {"base": "features", "extra": "features_extra", "v3": "features_v3", "all": "features_all"}[mode]
    cols = {"base": FEATURES, "extra": FEATURES_EXTRA, "v3": FEATURES_V3, "all": FEATURES_S1}[mode]
    fdir = os.path.join(run_dir, sub); os.makedirs(fdir, exist_ok=True)
    n_workers = n_workers or _default_workers()
    meta = os.path.join(fdir, "meta.json")
    if os.path.exists(meta):
        chunk = json.load(open(meta))["chunk"]
    elif any(f.startswith("part_") for f in os.listdir(fdir)):
        chunk = 1_000_000                                   # parts written by module v1.0
    else:
        json.dump({"chunk": chunk, "n": len(C)}, open(meta, "w"))
    starts = list(range(0, len(C), chunk))
    fn = lambda k: os.path.join(fdir, f"part_{k:04d}.parquet")
    jobs = [(k, np.arange(st, min(st + chunk, len(C))), mode) for k, st in enumerate(starts) if not os.path.exists(fn(k))]
    if jobs:
        _setup_features(C, S, res)
        t0 = time.time(); done = len(starts) - len(jobs)
        for k, M in _chunk_iter(jobs, n_workers):
            pd.DataFrame(M, columns=cols).to_parquet(fn(k), index=False); done += 1
            log(f"[features:{mode}] part {done}/{len(starts)} ({time.time() - t0:.0f}s)")
        _FX.clear(); gc.collect()
    return pd.concat([pd.read_parquet(fn(k)) for k in range(len(starts))], ignore_index=True)


def features_for_rows(rows, n_workers=None, chunk=500_000):
    """FEATURES_S1 matrix for a subset of candidate rows. Caller must have already called
    _setup_features(C, S, res) - this just reuses whatever is already in _FX."""
    n_workers = n_workers or _default_workers()
    jobs = [(k, rows[s:s + chunk], "all") for k, s in enumerate(range(0, len(rows), chunk))]
    out = [None] * len(jobs)
    for k, M in _chunk_iter(jobs, n_workers):
        out[k] = M
    return np.vstack(out) if out else np.zeros((0, len(FEATURES_S1)), np.float32)


def predict_stage1_streaming(C, S, res, model, out_dir, chunk=500_000, n_workers=None):
    """Test set: FEATURES_S1 + stage-1 prediction chunk by chunk; only probabilities are kept on disk."""
    pdir = os.path.join(out_dir, "pred1"); os.makedirs(pdir, exist_ok=True)
    n_workers = n_workers or _default_workers()
    starts = list(range(0, len(C), chunk))
    fn = lambda k: os.path.join(pdir, f"p_{k:04d}.npy")
    jobs = [(k, np.arange(st, min(st + chunk, len(C))), "all") for k, st in enumerate(starts) if not os.path.exists(fn(k))]
    if jobs:
        _setup_features(C, S, res)
        t0 = time.time(); done = len(starts) - len(jobs)
        for k, M in _chunk_iter(jobs, n_workers):
            np.save(fn(k), model.predict(M).astype(np.float32)); done += 1
            if done % 10 == 0 or done == len(starts):
                log(f"[predict stage 1] part {done}/{len(starts)} ({time.time() - t0:.0f}s)")
        _FX.clear(); gc.collect()
    return np.concatenate([np.load(fn(k)) for k in range(len(starts))])


# ---------- stage 2: sibling features ----------
def _jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def sibling_features(C, S, p1, min_p=0.05, conf=0.5):
    """For candidate rows with stage-1 p1 >= min_p: model-competition features + similarity to the S1's other
    confident candidates (p1 >= conf). Returns (row indices, float32 matrix of SIB_FEATURES)."""
    t0 = time.time()
    i1 = C.i1.values.astype(np.int64); i2 = C.i2.values.astype(np.int64)
    p1 = p1.astype(np.float32)
    d = pd.DataFrame({"i1": i1, "i2": i2, "p": p1})
    g1, g2 = d.groupby("i1").p, d.groupby("i2").p
    r1 = g1.rank(method="first", ascending=False).values - 1
    gap1 = g1.transform("max").values - p1
    r2 = g2.rank(method="first", ascending=False).values - 1
    mx2 = g2.transform("max").values
    gap2 = mx2 - p1
    order = np.lexsort((-p1, i2)); i2o = i2[order]
    first = np.r_[True, i2o[1:] != i2o[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    second = np.zeros(len(order), np.float32)
    has2 = np.r_[i2o[1:] == i2o[:-1], False] & first
    second[first] = 0
    second[np.flatnonzero(has2)] = p1[order][np.flatnonzero(has2) + 1]
    second_of_group = second[grp_start]
    other_best = np.where(first, second_of_group, p1[order][grp_start])
    ob = np.empty(len(p1), np.float32); ob[order] = other_best
    del d, g1, g2

    rows = np.flatnonzero(p1 >= min_p)
    rows = rows[np.lexsort((rows, i1[rows]))]
    T, W, N = S.ftoks.values if "ftoks" in S.columns else S.toks.values, S.words.values, S.nums.values
    M = np.zeros((len(rows), len(SIB_FEATURES)), np.float32)
    M[:, 0] = p1[rows]; M[:, 1] = r1[rows]; M[:, 2] = gap1[rows]; M[:, 3] = r2[rows]
    M[:, 4] = gap2[rows]; M[:, 5] = ob[rows]

    # v2.1 sib_addr_conflict: this row's S2/S3 record has an ALMOST IDENTICAL address (word set +
    # number set) to a record that is a DIFFERENT S1's rank-1, confident (p1 >= conf) candidate.
    # This is the "same building, two different real businesses" false-merge pattern (e.g. Kestari
    # Rain Inc vs Kestaria Inc-Services at the same street address) - distinct from sib_addr_jacc
    # below, which only compares a row to OTHER candidates of its OWN S1.
    strong = np.flatnonzero((p1 >= conf) & (r1 == 0))
    addr_key = {}
    for j in strong:
        b = i2[j]
        key = (W[b], N[b])
        if key[0] or key[1]:
            prev = addr_key.get(key)
            if prev is None or p1[j] > prev[1]:
                addr_key[key] = (i1[j], p1[j])
    conflict = np.zeros(len(rows), np.float32)
    for pos, j in enumerate(rows):
        b = i2[j]
        key = (W[b], N[b])
        if not (key[0] or key[1]):
            continue
        claim = addr_key.get(key)
        if claim is not None and claim[0] != i1[j]:
            conflict[pos] = claim[1]
    M[:, 12] = conflict
    # v3.0 b_conf_elsewhere: how many OTHER S1 hold this S2/S3 record at p1 >= conf
    cf_all = (p1 >= conf).astype(np.int32)
    ncf2 = pd.Series(cf_all).groupby(i2).transform("sum").values
    M[:, 15] = (ncf2 - cf_all)[rows]

    from rapidfuzz import fuzz
    gi = i1[rows]
    bounds = np.r_[0, np.flatnonzero(gi[1:] != gi[:-1]) + 1, len(rows)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        grp = rows[s:e]
        cf = [j for j in range(e - s) if p1[grp[j]] >= conf]
        if not cf:
            continue
        info = {}
        for j in range(e - s):
            b = i2[grp[j]]
            nm = T[b]
            info[j] = (set(nm.split()), fold(nm.replace(" ", "")), set(W[b].split()), N[b], nm)
        for j in range(e - s):
            others = [o for o in cf if o != j]
            if not others:
                continue
            nt, gl, ws, ns, nm = info[j]
            M[s + j, 6] = len(others)
            M[s + j, 7] = max(p1[grp[o]] for o in others)
            M[s + j, 8] = max(_jac(nt, info[o][0]) for o in others)
            M[s + j, 9] = float(bool(gl) and any(gl == info[o][1] for o in others))
            M[s + j, 10] = max((_jac(ws, info[o][2]) for o in others if ws and info[o][2]), default=0.0)
            M[s + j, 11] = float(bool(ns) and any(ns == info[o][3] for o in others))
            M[s + j, 13] = max((fuzz.token_set_ratio(nm, info[o][4]) for o in others if nm and info[o][4]),
                               default=0.0) / 100.0
            M[s + j, 14] = float(bool(ws) and any(ws == info[o][2] and ns == info[o][3] for o in others))
    log(f"[siblings] {len(rows):,} rows with p1 >= {min_p} ({time.time() - t0:.0f}s)")
    return rows, M


# =====================================================================================
# 9. MODEL, DECISION, SCORING
# =====================================================================================
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=42)


def _lgb_params(features, extra=None):
    p = dict(LGB_PARAMS)
    p["monotone_constraints"] = [MONO.get(f, 0) for f in features]
    p["monotone_constraints_method"] = "intermediate"
    p.update(extra or {})
    return p


def label_pairs(C, S, gt_pairs):
    """y = 1 if (S1, S2/S3) pair is in ground truth."""
    idx = pd.Index(S.entity_id.values)
    g = gt_pairs[gt_pairs.s1_id.isin(idx) & gt_pairs.m_id.isin(idx)]
    code_t = idx.get_indexer(g.s1_id).astype(np.int64) * (1 << 26) + idx.get_indexer(g.m_id)
    code_c = C.i1.values.astype(np.int64) * (1 << 26) + C.i2.values
    return np.isin(code_c, code_t).astype(np.int8)


def make_folds(groups, n_folds=3, seed=42):
    ug = np.unique(groups)
    f = np.random.RandomState(seed).randint(0, n_folds, len(ug))
    return f[np.searchsorted(ug, groups)].astype(np.int8)


def cross_validate(X, y, groups, n_folds=3, num_rounds=400, seed=42, params=None, features=None, folds=None):
    """Grouped K-fold by S1 -> out-of-fold probabilities for every row."""
    import lightgbm as lgb
    features = features or FEATURES
    fold = folds if folds is not None else make_folds(groups, n_folds, seed)
    oof = np.zeros(len(y), np.float32)
    imps = np.zeros(X.shape[1])
    for f in range(int(fold.max()) + 1):
        t0 = time.time()
        tr, va = fold != f, fold == f
        if va.sum() == 0:
            continue
        m = lgb.train(_lgb_params(features, params), lgb.Dataset(X[tr], label=y[tr], feature_name=features,
                                                                 free_raw_data=True), num_boost_round=num_rounds)
        oof[va] = m.predict(X[va])
        imps += m.feature_importance("gain")
        log(f"[cv] fold {f + 1}/{int(fold.max()) + 1}: train {tr.sum():,}, predict {va.sum():,} ({time.time() - t0:.0f}s)")
        del m; gc.collect()
    return oof, fold, pd.Series(imps, index=features).sort_values(ascending=False)


def s1_folds(S, n_folds=3, seed=42):
    """Fold id per ROW OF S (index it with i1). Stable across C, E and C+E, unlike make_folds."""
    return np.random.RandomState(seed).randint(0, n_folds, len(S)).astype(np.int8)


def cv_train(X, y, folds, groups, features, max_rounds=1500, early_stop=100, es_frac=0.1, seed=42,
             params=None, model_dir=None, tag="s1"):
    """Grouped CV with early stopping on an inner 10% group split of each training fold.
    Fold models are saved to model_dir (resumable) and returned. Returns dict(oof, models, iters, imp)."""
    import lightgbm as lgb
    nf = int(folds.max()) + 1
    oof = np.zeros(len(y), np.float32); imps = np.zeros(len(features)); models, iters = [], []
    rng = np.random.RandomState(seed)
    for f in range(nf):
        t0 = time.time()
        fp = os.path.join(model_dir, f"{tag}_fold{f}.txt") if model_dir else None
        va = np.flatnonzero(folds == f)
        if fp and os.path.exists(fp):
            m = lgb.Booster(model_file=fp)
        else:
            tr = np.flatnonzero(folds != f)
            ug = np.unique(groups[tr])
            es = np.isin(groups[tr], ug[rng.rand(len(ug)) < es_frac])
            dtr = lgb.Dataset(X[tr[~es]], label=y[tr[~es]], feature_name=features, free_raw_data=True)
            des = lgb.Dataset(X[tr[es]], label=y[tr[es]], reference=dtr, free_raw_data=True)
            del tr, es
            m = lgb.train({**_lgb_params(features, params), "metric": "binary_logloss"}, dtr,
                          num_boost_round=max_rounds, valid_sets=[des],
                          callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(250)])
            del dtr, des; gc.collect()
            if fp:
                m.save_model(fp, num_iteration=m.best_iteration or None)
        it = m.best_iteration if (m.best_iteration or 0) > 0 else m.current_iteration()
        oof[va] = m.predict(X[va], num_iteration=it)
        imps += m.feature_importance("gain", iteration=it)
        models.append(m); iters.append(int(it))
        log(f"[cv {tag}] fold {f + 1}/{nf}: best_iter {it}, predict {len(va):,} ({time.time() - t0:.0f}s)")
        gc.collect()
    return {"oof": oof, "models": models, "iters": iters,
            "imp": pd.Series(imps, index=features).sort_values(ascending=False)}


def realistic_mask(C, S, gt_pairs):
    """TRAIN/VALIDATION ONLY. False for candidate rows whose S2/S3 record is an 'orphan' - its true S1 is
    not in the selection (so it can only ever be a wrong match here, while on test its true S1 is always
    present). Labels choose which rows to train/score on; they never enter a feature."""
    idx = pd.Index(S.entity_id.values)
    orphan_ids = gt_pairs.m_id.values[~gt_pairs.s1_id.isin(idx).values]
    orph = np.zeros(len(S), bool)
    pos = idx.get_indexer(orphan_ids); orph[pos[pos >= 0]] = True
    keep = ~orph[C.i2.values.astype(np.int64)]
    log(f"[realistic] orphan S2/S3 records in selection: {int(orph.sum()):,} | candidate rows dropped: "
        f"{int((~keep).sum()):,} of {len(C):,}")
    return keep


def assign_best(i1, i2, p):
    """Each S2/S3 record keeps only its highest-probability S1."""
    order = np.lexsort((-p, i2))
    first = np.r_[True, i2[order][1:] != i2[order][:-1]]
    best = np.zeros(len(p), bool); best[order[first]] = True
    return best


def macro_f05(s1_rows, true_i1, true_i2, pred_i1, pred_i2):
    """Exact leaderboard metric: per-S1 F0.5, averaged over ALL S1 (singletons included)."""
    s1_rows = np.asarray(s1_rows)
    pos = pd.Series(np.arange(len(s1_rows)), index=s1_rows)
    nt = np.bincount(pos.reindex(true_i1).values, minlength=len(s1_rows)).astype(float)
    npd = np.bincount(pos.reindex(pred_i1).values, minlength=len(s1_rows)).astype(float)
    ct = true_i1.astype(np.int64) * (1 << 26) + true_i2
    cp = pred_i1.astype(np.int64) * (1 << 26) + pred_i2
    hit = np.isin(cp, ct)
    tp = np.bincount(pos.reindex(pred_i1[hit]).values, minlength=len(s1_rows)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npd > 0, tp / npd, 0.0); Rr = np.where(nt > 0, tp / nt, 0.0)
        f = np.where((P + Rr) > 0, 1.25 * P * Rr / (0.25 * P + Rr), 0.0)
    f = np.where(nt == 0, (npd == 0).astype(float), f)
    return f, nt, npd, tp


def evaluate(C, p, S, gt_pairs, taus=None):
    """Tune threshold for macro F0.5 and report metrics vs ground truth.
    v2.1: also tunes and reports a PER-COUNTRY threshold (rep['tau_by_country'],
    rep['F05_percountry_tau']) alongside the original single global tau. The returned `m` (kept
    pairs) and every other value are UNCHANGED from v2.0 - this is purely additional reporting so
    existing callers (and the global-tau behavior itself) are unaffected. Use rep['tau_by_country']
    with decide()'s dict form only after confirming F05_percountry_tau >= F05 on your own validation
    run; if it's ever lower (e.g. a country has very few S1 rows and the per-country tau overfits),
    fall back to the single global tau."""
    taus = taus if taus is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)
    idx = pd.Index(S.entity_id.values)
    s1_rows = np.flatnonzero((S.src == "S1").values)
    g = gt_pairs[gt_pairs.s1_id.isin(idx)]
    ti1 = idx.get_indexer(g.s1_id).astype(np.int64)
    ti2 = idx.get_indexer(g.m_id).astype(np.int64)        # -1 = match outside selection: counts as a miss
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    best = assign_best(i1, i2, p)
    rows = []
    for t in taus:
        m = best & (p >= t)
        f, nt, npd, tp = macro_f05(s1_rows, ti1, ti2, i1[m], i2[m])
        rows.append((t, f.mean()))
    grid = pd.DataFrame(rows, columns=["tau", "F05"])
    tau = float(grid.loc[grid.F05.idxmax(), "tau"])
    m = best & (p >= tau)
    f, nt, npd, tp = macro_f05(s1_rows, ti1, ti2, i1[m], i2[m])
    ctry = S.country.astype(str).values[s1_rows]
    cand_codes = i1 * (1 << 26) + i2
    true_codes = ti1 * (1 << 26) + ti2
    fperf, *_ = macro_f05(s1_rows, ti1, ti2, i1[np.isin(cand_codes, true_codes)], i2[np.isin(cand_codes, true_codes)])

    # per-country tau: same grid search, restricted to each country's own S1 rows and candidates.
    # macro_f05 requires every true_i1/pred_i1 value to be a member of the s1_rows subset it's given
    # (it reindexes positions against it), so true pairs whose S1 sits in a DIFFERENT country must be
    # dropped from ti1c/ti2c below - they can't be predicted correctly from within this subset anyway,
    # and leaving them in would crash the bincount reindex with an out-of-subset index.
    cand_ctry = S.country.astype(str).values[i1]
    tau_by_country = {}
    for c in np.unique(ctry):
        s1_c = s1_rows[ctry == c]
        s1_c_set = set(s1_c.tolist())
        keep_t = np.fromiter((x in s1_c_set for x in ti1), bool, len(ti1))
        ti1c, ti2c = ti1[keep_t], ti2[keep_t]
        cm = cand_ctry == c
        best_f, best_t = -1.0, tau
        for t in taus:
            mt = best & (p >= t) & cm
            fc, *_ = macro_f05(s1_c, ti1c, ti2c, i1[mt], i2[mt])
            if fc.mean() > best_f:
                best_f, best_t = fc.mean(), float(t)
        tau_by_country[c] = best_t
    m_pc = np.zeros(len(p), bool)
    for c, t in tau_by_country.items():
        cm = cand_ctry == c
        m_pc |= best & (p >= t) & cm
    f_pc, *_ = macro_f05(s1_rows, ti1, ti2, i1[m_pc], i2[m_pc])

    rep = {
        "n_s1": int(len(s1_rows)), "n_true_pairs": int(len(ti1)), "n_candidates": int(len(C)),
        "cand_per_s1": float(len(C) / max(len(s1_rows), 1)),
        "blocking_recall": float(np.isin(true_codes, cand_codes).mean()) if len(ti1) else 1.0,
        "F05_perfect_model_on_candidates": float(fperf.mean()),
        "tau": tau, "F05": float(f.mean()),
        "F05_at_0.5": float(grid.loc[(grid.tau - 0.5).abs().idxmin(), "F05"]),
        "micro_precision": float(tp.sum() / max(npd.sum(), 1)), "micro_recall": float(tp.sum() / max(nt.sum(), 1)),
        "singleton_share": float((nt == 0).mean()),
        "singleton_accuracy": float(((nt == 0) & (npd == 0)).sum() / max((nt == 0).sum(), 1)),
        "F05_matched_s1": float(f[nt > 0].mean()) if (nt > 0).any() else None,
        "F05_by_country": {c: float(f[ctry == c].mean()) for c in np.unique(ctry)},
        "tau_by_country": tau_by_country, "F05_percountry_tau": float(f_pc.mean()),
    }
    return rep, grid, m, (s1_rows, f, nt, npd, tp)


# ---------- v3.0 decision: expected-F0.5 per S1 ----------
def gt_arrays(S, gt_pairs):
    idx = pd.Index(S.entity_id.values)
    g = gt_pairs[gt_pairs.s1_id.isin(idx)]
    return {"s1_rows": np.flatnonzero((S.src == "S1").values),
            "ti1": idx.get_indexer(g.s1_id).astype(np.int64), "ti2": idx.get_indexer(g.m_id).astype(np.int64),
            "ctry": S.country.astype(str).values}


def score_keep(C, keep, G):
    """Macro F0.5 (+ per-country) of a kept-pair mask."""
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    f, nt, npd, tp = macro_f05(G["s1_rows"], G["ti1"], G["ti2"], i1[keep], i2[keep])
    c = G["ctry"][G["s1_rows"]]
    return {"F05": float(f.mean()), "micro_precision": float(tp.sum() / max(npd.sum(), 1)),
            "micro_recall": float(tp.sum() / max(nt.sum(), 1)),
            "singleton_accuracy": float(((nt == 0) & (npd == 0)).sum() / max((nt == 0).sum(), 1)),
            "F05_by_country": {k: float(f[c == k].mean()) for k in np.unique(c)}}


def expected_f_keep(C, p, gamma=1.0, lam=0.0, miss=0.0, pmin=0.01, kmax=40):
    """For each S1: sort its (assign_best) candidates by q=p**gamma and keep the top k that maximises
    E[F0.5] ~ 1.25*sum_{i<=k} q_i / (0.25*(1 + Q - q_i + miss) + k)   (Q = sum of all q of that S1,
    miss = expected true matches outside the candidates), vs E[F0.5 | predict nothing] = prod(1-q_i).
    lam > 0 favours predicting nothing (precision), lam < 0 favours predicting."""
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    best = assign_best(i1, i2, p)
    sel = np.flatnonzero(best & (p >= pmin))
    keep = np.zeros(len(p), bool)
    if len(sel) == 0:
        return keep
    q = np.clip(p[sel].astype(np.float64), 0, 1) ** gamma
    order = np.lexsort((-q, i1[sel])); rows = sel[order]; qs = q[order]; g = i1[rows]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]; sizes = np.diff(np.r_[starts, len(g)])
    G = len(starts); gid = np.repeat(np.arange(G), sizes); rank = np.arange(len(g)) - np.repeat(starts, sizes)
    Q = np.add.reduceat(qs, starts)
    E0 = np.exp(np.add.reduceat(np.log1p(-np.minimum(qs, 1 - 1e-7)), starts))
    K = int(min(kmax, sizes.max()))
    inK = rank < K
    Qm = np.zeros((G, K), np.float32); Qm[gid[inK], rank[inK]] = qs[inK]
    Cm = (0.25 * (1.0 + Q[:, None] - Qm + miss)).astype(np.float32)
    bestE = np.full(G, -np.inf, np.float32); bestk = np.zeros(G, np.int32)
    for k in range(1, K + 1):
        Ek = 1.25 * (Qm[:, :k] / (Cm[:, :k] + k)).sum(1)
        upd = (sizes >= k) & (Ek > bestE)
        bestE[upd] = Ek[upd]; bestk[upd] = k
    kk = np.where(E0 + lam >= bestE, 0, bestk)
    keep[rows[rank < kk[gid]]] = True
    return keep


def tune_decision(C, p, S, gt_pairs, gammas=(0.8, 1.0, 1.25, 1.5, 2.0), lams=(-0.1, -0.05, 0.0, 0.05, 0.1, 0.2),
                  misses=(0.0, 0.1, 0.25), taus=None):
    """Grid-search the expected-F rule AND the old global threshold on the same validation predictions;
    returns (best decision dict, grid DataFrame). The decision dict feeds apply_decision on test."""
    G = gt_arrays(S, gt_pairs); rows = []
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    best = assign_best(i1, i2, p)
    for t in (taus if taus is not None else np.round(np.arange(0.3, 0.91, 0.02), 2)):
        rows.append({"method": "threshold", "tau": float(t), "F05": score_keep(C, best & (p >= t), G)["F05"]})
    for gm in gammas:
        for ms in misses:
            for lm in lams:
                k = expected_f_keep(C, p, gm, lm, ms)
                rows.append({"method": "expected", "gamma": gm, "lam": lm, "miss": ms, "F05": score_keep(C, k, G)["F05"]})
    grid = pd.DataFrame(rows).sort_values("F05", ascending=False).reset_index(drop=True)
    b = grid.iloc[0].to_dict()
    dec = {"method": b["method"], "F05": b["F05"]}
    if b["method"] == "threshold":
        dec["tau"] = b["tau"]
    else:
        dec.update(gamma=b["gamma"], lam=b["lam"], miss=b["miss"])
    th = grid[grid.method == "threshold"].iloc[0]
    dec["threshold_fallback"] = {"tau": float(th["tau"]), "F05": float(th["F05"])}
    return dec, grid


def apply_decision(C, p, dec, S=None):
    if dec["method"] == "threshold":
        return decide(C, p, dec["tau"], S)
    return expected_f_keep(C, p, dec["gamma"], dec["lam"], dec["miss"])


# ---------- v3.0 candidate expansion ----------
def _rec_keys_v3(rows, cty, ft, W, N, chunk=1_000_000):
    """(row, key hash, key type) for exact name-set (0), glued name (1), full address (2) keys.
    Built in chunks so only hashes (not key strings) are held for the full test set."""
    if len(rows) > chunk:
        return pd.concat([_rec_keys_v3(rows[s:s + chunk], cty, ft, W, N, chunk)
                          for s in range(0, len(rows), chunk)], ignore_index=True)
    R, K, T = [], [], []
    for r in rows:
        c, t = cty[r], ft[r]
        if t:
            toks = t.split()
            ns = " ".join(sorted(set(toks)))
            if len(ns) >= 4:
                R.append(r); K.append(f"{c}|N|{ns}"); T.append(0)
            gl = fold("".join(toks))
            if len(gl) >= 6:
                R.append(r); K.append(f"{c}|G|{gl}"); T.append(1)
        if W[r] and N[r]:
            R.append(r); K.append(f"{c}|A|{' '.join(sorted(set(W[r].split())))}#{' '.join(sorted(set(N[r].split())))}")
            T.append(2)
    h = pd.util.hash_array(np.array(K, dtype=object)) if K else np.zeros(0, np.uint64)
    return pd.DataFrame({"r": np.array(R, np.int64), "key": h, "kt": np.array(T, np.int8)})


def _pair_sims(E, S):
    """sn (name char-3gram TF-IDF cosine) / sa (address token TF-IDF cosine) for new pairs, fitted per
    country on the records involved (same recipe as blocking, different corpus - same on train and test)."""
    i1, i2 = E.i1.values.astype(np.int64), E.i2.values.astype(np.int64)
    sn, sa = np.zeros(len(E), np.float32), np.zeros(len(E), np.float32)
    cty = S.country.astype(str).values
    T, W, N = S.toks.values, S.words.values, S.nums.values
    for c in np.unique(cty[i1]):
        m = np.flatnonzero(cty[i1] == c)
        recs = np.unique(np.r_[i1[m], i2[m]])
        la, lb = np.searchsorted(recs, i1[m]), np.searchsorted(recs, i2[m])
        Xn = _tfidf([T[r] for r in recs], analyzer="char_wb", ngram_range=(3, 3))
        Xa = _tfidf([f"{W[r]} {N[r]}".strip() for r in recs], tokenizer=str.split, token_pattern=None,
                    lowercase=False)
        sn[m] = _rowdot(Xn, la, lb); sa[m] = _rowdot(Xa, la, lb)
    return sn, sa


def expand_candidates(C, S, p1, conf=0.7, max_key=20, max_new=20):
    """New (S1, S2/S3) pairs NOT already in C: an S2/S3 record sharing a non-generic (<= max_key records)
    exact name-set / glued-name / full-address key with the S1 itself or with one of the S1's confident
    matches (assign_best & p1 >= conf). Label-free. Returns E with blocking columns + exp_sib/exp_kaddr."""
    t0 = time.time()
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    strong = np.flatnonzero(assign_best(i1, i2, p1) & (p1 >= conf))
    cty = S.country.astype(str).values
    ft = S.ftoks.values if "ftoks" in S.columns else S.toks.values
    W, N = S.words.values, S.nums.values
    is1 = (S.src == "S1").values
    K23 = _rec_keys_v3(np.flatnonzero(~is1), cty, ft, W, N).rename(columns={"r": "c"})
    vc = K23.key.value_counts()
    K23 = K23[K23.key.isin(vc.index[vc.values <= max_key])][["key", "c"]]
    seeds = pd.DataFrame({"a": np.r_[i1[strong], np.flatnonzero(is1)],
                          "b": np.r_[i2[strong], np.flatnonzero(is1)],
                          "via": np.r_[np.ones(len(strong), np.int8), np.zeros(int(is1.sum()), np.int8)]})
    Kb = _rec_keys_v3(np.unique(seeds.b.values), cty, ft, W, N).rename(columns={"r": "b"})
    M = seeds.merge(Kb, on="b")[["a", "key", "kt", "via"]].merge(K23, on="key")
    del Kb, K23, seeds; gc.collect()
    code = M.a.values * (1 << 26) + M.c.values
    ccode = np.sort(i1 * (1 << 26) + i2)
    pos = np.clip(np.searchsorted(ccode, code), 0, max(len(ccode) - 1, 0))
    M = M[~((len(ccode) > 0) & (ccode[pos] == code))]
    M = M.assign(kad=(M.kt.values == 2).astype(np.int8))
    M = M.groupby(["a", "c"], sort=False).agg(via=("via", "min"), kad=("kad", "max")).reset_index()
    M = M.sort_values(["a", "kad", "via"], ascending=[True, False, True])
    M = M[M.groupby("a").cumcount() < max_new].reset_index(drop=True)
    is_pool = S.part.astype(str).str.endswith("|POOL").to_numpy(dtype=bool)
    E = pd.DataFrame({"i1": M.a.values.astype(np.int64), "i2": M.c.values.astype(np.int64),
                      "bits": np.full(len(M), 1 << 5, np.int16), "knn": np.zeros(len(M), bool)})
    E["sn"], E["sa"] = _pair_sims(E, S) if len(E) else (np.zeros(0, np.float32), np.zeros(0, np.float32))
    E["pool"] = is_pool[E.i2.values]
    E["exp_sib"] = M.via.values.astype(np.int8); E["exp_kaddr"] = M.kad.values.astype(np.int8)
    log(f"[expand] seeds: {len(strong):,} confident pairs + {int(is1.sum()):,} S1 | new pairs {len(E):,} "
        f"(via sibling only {int(E.exp_sib.sum()):,}, address key {int(E.exp_kaddr.sum()):,}) "
        f"({time.time() - t0:.0f}s)")
    return E


def concat_candidates(C, E):
    cols = ["i1", "i2", "bits", "knn", "sn", "sa", "pool", "exp_sib", "exp_kaddr"]
    a = C.reindex(columns=cols); a["exp_sib"] = a.exp_sib.fillna(0); a["exp_kaddr"] = a.exp_kaddr.fillna(0)
    out = pd.concat([a, E[cols]], ignore_index=True)
    return out.astype({"i1": np.int64, "i2": np.int64, "bits": np.int16, "knn": bool, "sn": np.float32,
                       "sa": np.float32, "pool": bool, "exp_sib": np.int8, "exp_kaddr": np.int8})


def features_for_new(C, E, S, res, n_workers=None):
    """FEATURES_S1 for the expansion rows E, with competition features computed over C + E."""
    CE = concat_candidates(C, E)
    _setup_features(CE, S, res)
    X = features_for_rows(np.arange(len(C), len(CE)), n_workers=n_workers)
    _FX.clear(); del CE; gc.collect()
    return X


def write_id_lists(path, header2, s1_ids, i1_rows, i2_rows, S):
    """TSV in the submission format: one row per S1 (empty list allowed)."""
    eid = S.entity_id.values
    lists = defaultdict(list)
    for a, b in zip(i1_rows, i2_rows):
        lists[a].append(eid[b])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{header2}\n")
        for r in s1_ids:
            fh.write(f"{eid[r]}\t{','.join(dict.fromkeys(lists.get(r, [])))}\n")


def write_comparison(path, S, C, keep, gt_pairs, detail):
    """Per S1: true ids, predicted ids, candidate count, F0.5 - for inspecting the validation run."""
    s1_rows, f, nt, npd, tp = detail
    eid = S.entity_id.values
    pred = defaultdict(list)
    for a, b in zip(C.i1.values[keep], C.i2.values[keep]):
        pred[a].append(eid[b])
    ncand = pd.Series(C.i1.values).value_counts()
    tr = gt_pairs[gt_pairs.s1_id.isin(set(eid[s1_rows]))].groupby("s1_id").m_id.apply(lambda x: ",".join(sorted(x)))
    out = pd.DataFrame({"source1_entity_id": eid[s1_rows], "country": S.country.astype(str).values[s1_rows],
                        "n_true": nt.astype(int), "n_pred": npd.astype(int), "n_correct": tp.astype(int),
                        "n_candidates": ncand.reindex(s1_rows).fillna(0).astype(int).values,
                        "f05": np.round(f, 4)})
    out["true_ids"] = out.source1_entity_id.map(tr).fillna("")
    out["predicted_ids"] = [",".join(sorted(pred.get(r, []))) for r in s1_rows]
    out.to_csv(path, sep="\t", index=False)
    return out


def raw_lookup(ids, split, data_dir, parquet_dir=None):
    """Original name/address for a set of entity ids (reads one raw source at a time)."""
    ids = set(ids); out = []
    for k in (1, 2, 3):
        df = load_raw(split, k, data_dir, parquet_dir)
        out.append(df[df.entity_id.isin(ids)][["entity_id", "business_name", "business_address"]])
        del df; gc.collect()
    return pd.concat(out).set_index("entity_id")


def show_errors(C, p, keep, S, y, data_dir, parquet_dir=None, n=12, seed=0):
    """Print false merges (precision errors) and missed matches (recall errors) with the raw text."""
    rng = np.random.RandomState(seed)
    fp = np.flatnonzero(keep & (y == 0)); fn = np.flatnonzero(~keep & (y == 1))
    fp = rng.choice(fp, min(n, len(fp)), replace=False) if len(fp) else fp
    fn = rng.choice(fn, min(n, len(fn)), replace=False) if len(fn) else fn
    eid = S.entity_id.values
    ids = set(eid[C.i1.values[np.r_[fp, fn]]]) | set(eid[C.i2.values[np.r_[fp, fn]]])
    raw = raw_lookup(ids, "train", data_dir, parquet_dir)
    for title, rows in [("FALSE MERGES (predicted, not true)", fp), ("MISSED MATCHES (true, not predicted)", fn)]:
        print("\n" + "#" * 25, title, "#" * 25)
        for r in rows:
            a, b = eid[C.i1.values[r]], eid[C.i2.values[r]]
            print(f"p={p[r]:.3f}  {a} | {raw.loc[a].business_name} || {raw.loc[a].business_address}")
            print(f"           {b} | {raw.loc[b].business_name} || {raw.loc[b].business_address}\n")



# =====================================================================================
# 10. FINAL MODEL + TEST SUBMISSION
# =====================================================================================
def train_final(X, y, features, num_rounds=400, params=None):
    """One model on all labelled rows (used for the test set)."""
    import lightgbm as lgb
    t0 = time.time()
    m = lgb.train(_lgb_params(features, params), lgb.Dataset(X, label=y, feature_name=features, free_raw_data=True),
                  num_boost_round=num_rounds)
    log(f"[final model] {len(features)} features, trained on {len(y):,} rows ({time.time() - t0:.0f}s)")
    return m


def decide(C, p, tau, S=None):
    """Each S2/S3 to its best S1 only, then keep pairs with probability >= tau.
    v2.1: tau may be a scalar (v2.0 behavior, unchanged) OR a {country: tau} dict (as produced by
    evaluate()'s rep['tau_by_country']) - in the dict case S (the same frame passed to evaluate/
    build_features) is required, to look up each candidate's country. Falls back to the dict's own
    values only for countries present in it; a country missing from the dict keeps ALL its candidates
    (fail-open on recall, never silently drops a country you forgot to tune)."""
    best = assign_best(C.i1.values.astype(np.int64), C.i2.values.astype(np.int64), p)
    if not isinstance(tau, dict):
        return best & (p >= tau)
    assert S is not None, "decide(): S is required when tau is a per-country dict"
    cand_ctry = S.country.astype(str).values[C.i1.values.astype(np.int64)]
    keep = np.zeros(len(p), bool)
    for c, t in tau.items():
        cm = cand_ctry == c
        keep |= best & (p >= t) & cm
    missing = ~np.isin(cand_ctry, list(tau.keys()))
    keep |= best & missing
    return keep


def write_submission(S, C, keep, out_dir):
    """matching_results.tsv + candidate_pairs.tsv with one row for EVERY S1 in S."""
    os.makedirs(out_dir, exist_ok=True)
    s1_rows = np.flatnonzero((S.src == "S1").values)
    mp_ = os.path.join(out_dir, "matching_results.tsv"); cp_ = os.path.join(out_dir, "candidate_pairs.tsv")
    write_id_lists(mp_, "matched_entity_ids", s1_rows, C.i1.values[keep], C.i2.values[keep], S)
    write_id_lists(cp_, "candidate_entity_ids", s1_rows, C.i1.values, C.i2.values, S)
    n_match = pd.Series(C.i1.values[keep]).nunique()
    log(f"[submission] S1 rows {len(s1_rows):,} | with >=1 match {n_match:,} "
        f"({100 * n_match / max(len(s1_rows), 1):.1f}%) | matched pairs {int(keep.sum()):,}")
    return mp_, cp_


# =====================================================================================
# 11. TEST: STAGE 2 OVER C + E  (v2.3 embedding section removed in v3.0)
# =====================================================================================
def predict_stage2(CE, S, res, p1, model2, out_dir, min_p=0.02, n_c=None, n_workers=None):
    """Test set: sibling features + FEATURES_S1 for rows with p1 >= min_p, stage-2 prediction.
    CE = concat_candidates(C, E); n_c = len(C). Blocking rows get their stage-1 features with competition
    over C only and expansion rows over C + E - exactly as in training. Cached in out_dir/pred2_v3.npz."""
    fp = os.path.join(out_dir, "pred2_v3.npz")
    if os.path.exists(fp):
        z = np.load(fp); p = p1.copy(); p[z["rows"]] = z["p2"]; return p
    n_c = len(CE) if n_c is None else n_c
    rows, M = sibling_features(CE, S, p1, min_p=min_p)
    X1 = np.zeros((len(rows), len(FEATURES_S1)), np.float32)
    rc = rows < n_c
    t0 = time.time()
    for msk, frame in ((rc, CE.iloc[:n_c]), (~rc, CE)):
        if msk.any():
            _setup_features(frame, S, res)
            X1[msk] = features_for_rows(rows[msk], n_workers=n_workers)
            _FX.clear(); gc.collect()
    log(f"[stage 2] features for {len(rows):,} rows ({time.time() - t0:.0f}s)")
    p2 = model2.predict(np.hstack([X1, M])).astype(np.float32)
    np.savez(fp, rows=rows, p2=p2)
    p = p1.copy(); p[rows] = p2
    return p

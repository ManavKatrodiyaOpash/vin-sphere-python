"""
DriveArabia UAE Toyota Price Scraper
=====================================
Scrapes all Toyota models, their trims, and prices from:
  https://www.drivearabia.com/carprices/uae/toyota/{model}/{year}/

Output CSV columns:
    model_slug, year, trim_name,
    price_min_aed, price_max_aed, price_avg_aed,
    price_raw, currency, url, scraped_at

Price logic:
    Range  (e.g. "AED 99,000 - AED 120,000") â†’ min + max + avg populated
    Single (e.g. "AED 245,000")               â†’ only avg populated

Usage:
    python scrape_drivearabia_toyota.py
    python scrape_drivearabia_toyota.py --start-year 2020 --end-year 2026
    python scrape_drivearabia_toyota.py --models toyota-land-cruiser toyota-camry
    python scrape_drivearabia_toyota.py --delay 2.0 --output my_prices.csv
    python scrape_drivearabia_toyota.py --debug-html  # dumps raw HTML per page for inspection
"""

import re
import csv
import time
import logging
import argparse
from pathlib import Path
from datetime import UTC, datetime

import requests
from bs4 import BeautifulSoup, Tag

# â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

BASE_URL     = "https://www.drivearabia.com"
TOYOTA_INDEX = f"{BASE_URL}/carprices/uae/toyota/"

DEFAULT_START_YEAR = 2022
DEFAULT_END_YEAR   = 2026
DEFAULT_DELAY      = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
    "Cache-Control":   "max-age=0",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# â”€â”€â”€ Known competitor makes (trim bleed filter) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_KNOWN_MAKES = re.compile(
    r"\b(hyundai|honda|nissan|ford|chevrolet|kia|mazda|bmw|mercedes|audi|"
    r"lexus|mitsubishi|jeep|dodge|ram|infiniti|volkswagen|volvo|subaru|"
    r"genesis|porsche|land rover|jaguar|renault|peugeot|fiat|suzuki|isuzu|"
    r"gmc|cadillac|buick|lincoln|acura|chrysler|skoda|seat|opel|citroen|"
    r"dacia|alfa romeo|maserati|ferrari|lamborghini|bentley|rolls royce|"
    r"aston martin|bugatti|mclaren|haval|geely|chery|mg|byd|great wall)\b",
    re.I,
)

# â”€â”€â”€ Trim name validator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_JUNK_WORDS = re.compile(
    r"\b(the|is|are|was|were|has|have|this|that|with|from|for|and|or|"
    r"view|compare|check|see|read|more|latest|new|used|buy|price|"
    r"review|photos|specs|variants|overview|starting|calculator|similar|"
    r"vehicle|brand|body|engine|fuel|weight|toyota)\b",
    re.I,
)


def is_valid_trim(text: str) -> bool:
    if not text or len(text) > 70:
        return False
    if _KNOWN_MAKES.search(text):
        return False
    if re.search(r"(https?://|www\.)", text, re.I):
        return False
    if _JUNK_WORDS.search(text):
        return False
    # reject if >40% of chars are digits (price leaked into trim)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.4:
        return False
    return True


def model_name_from_slug(model_slug: str) -> str:
    parts = [
        p
        for p in model_slug.replace("toyota-", "").replace("-", " ").split()
        if p
    ]
    return " ".join(parts)


def clean_trim_name(text: str, model_slug: str = "") -> str:
    """
    Keep only the trim label, not page labels, prices, links, or model prefixes.
    Examples:
      "Avalon Limited" -> "Limited"
      "2.5L I4 E FWDAED 109,900 - 110,000" -> "2.5L I4 E FWD"
      "Toyota Camry XLE" -> "XLE"
    """
    if not text:
        return ""

    trim = re.sub(r"\s+", " ", str(text)).strip()
    trim = re.split(r"AED", trim, maxsplit=1, flags=re.I)[0].strip()
    trim = re.sub(r"\b(Contact Dealer|See Similar Cars|Check Used Price)\b.*$", "", trim, flags=re.I).strip()
    trim = re.sub(r"^[\W_]+|[\W_]+$", "", trim)

    model_name = model_name_from_slug(model_slug)
    if model_name:
        # Remove "Toyota <model>" or "<model>" only when a trim remains.
        model_pattern = re.escape(model_name).replace(r"\ ", r"\s+")
        patterns = [
            rf"^toyota\s+{model_pattern}\s+(.+)$",
            rf"^{model_pattern}\s+(.+)$",
        ]
        for pattern in patterns:
            m = re.match(pattern, trim, re.I)
            if m and m.group(1).strip():
                trim = m.group(1).strip()
                break

    return re.sub(r"\s+", " ", trim).strip()


# â”€â”€â”€ Price parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_PRICE_NUM_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{4,8})\b")
_RANGE_SEP_RE = re.compile(r"\s*(?:–|—|â€“|â€”|-|to)\s*", re.I)
_AED_RE       = re.compile(r"AED\s*[\d,]+", re.I)


def _extract_nums(raw: str) -> list[int]:
    nums = []
    for m in _PRICE_NUM_RE.finditer(raw):
        digits = re.sub(r"[^\d]", "", m.group(1))
        if 4 <= len(digits) <= 8:
            nums.append(int(digits))
    return nums


def parse_prices(raw: str) -> dict:
    """
    Returns price_min_aed, price_max_aed, price_avg_aed.
    Range  â†’ all three populated.
    Single â†’ only price_avg_aed populated.
    """
    empty = {"price_min_aed": None, "price_max_aed": None, "price_avg_aed": None}
    if not raw:
        return empty

    # Try range split
    halves = _RANGE_SEP_RE.split(raw, maxsplit=1)
    if len(halves) == 2:
        l = _extract_nums(halves[0])
        r = _extract_nums(halves[1])
        if l and r:
            lo, hi = min(l[0], r[0]), max(l[0], r[0])
            if lo != hi:
                return {
                    "price_min_aed": lo,
                    "price_max_aed": hi,
                    "price_avg_aed": round((lo + hi) / 2),
                }

    # Single price
    nums = _extract_nums(raw)
    if not nums:
        return empty
    return {"price_min_aed": None, "price_max_aed": None, "price_avg_aed": nums[0]}


# â”€â”€â”€ Session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # warm up cookies
    try:
        s.get(BASE_URL, timeout=10)
    except Exception:
        pass
    return s


def fetch(session: requests.Session, url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=15,
                               headers={"Referer": TOYOTA_INDEX})
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                log.debug("404 %s", url)
                return None
            elif resp.status_code == 429:
                wait = 15 * attempt
                log.warning("Rate-limited. Sleeping %ds", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d â€” %s (attempt %d)", resp.status_code, url, attempt)
        except requests.RequestException as e:
            log.warning("Request error %s (attempt %d): %s", url, attempt, e)
            time.sleep(3 * attempt)
    return None


# â”€â”€â”€ HTML debug dump â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def dump_page_structure(soup: BeautifulSoup, slug: str, year: int) -> None:
    """
    Prints a structured summary of the page to help diagnose parser misses.
    Activate with --debug-html flag.
    """
    print(f"\n{'='*70}")
    print(f"DEBUG DUMP: {slug} / {year}")
    print(f"{'='*70}")

    tables = soup.find_all("table")
    print(f"\n[TABLES: {len(tables)}]")
    for i, t in enumerate(tables):
        print(f"  Table {i}:")
        for tr in t.find_all("tr")[:5]:
            cells = [td.get_text(strip=True)[:40] for td in tr.find_all(["td", "th"])]
            print(f"    {cells}")

    aed_nodes = soup.find_all(string=_AED_RE)
    print(f"\n[AED TEXT NODES: {len(aed_nodes)}]")
    for n in aed_nodes[:20]:
        p = n.parent
        gp = p.parent if p else None
        print(f"  text={repr(n.strip()[:60])}")
        print(f"    parent  â†’ <{p.name}> class={p.get('class')} text={p.get_text(strip=True)[:60]}")
        if gp:
            print(f"    grandp  â†’ <{gp.name}> class={gp.get('class')} text={gp.get_text(strip=True)[:60]}")

    price_els = soup.find_all(
        ["div", "span", "li", "td", "p"],
        class_=re.compile(r"price|trim|variant|spec|grade|version|car[-_]|row|item", re.I),
    )
    print(f"\n[PRICE/TRIM CLASS ELEMENTS: {len(price_els)}]")
    for el in price_els[:20]:
        print(f"  <{el.name}> class={el.get('class')} â†’ {el.get_text(strip=True)[:80]}")

    print(f"\n[ALL UL/OL LISTS (first 3)]")
    for ul in soup.find_all(["ul", "ol"])[:3]:
        for li in ul.find_all("li")[:6]:
            print(f"  <li> {li.get_text(strip=True)[:80]}")

    print(f"{'='*70}\n")


# â”€â”€â”€ Core parser: tries 5 strategies in order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_trims(
    soup: BeautifulSoup,
    model_slug: str,
    year: int,
    url: str,
    debug_html: bool = False,
) -> list[dict]:

    if debug_html:
        dump_page_structure(soup, model_slug, year)

    rows = []
    now  = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def make_row(trim_name: str, price_raw: str) -> dict:
        trim_name = clean_trim_name(trim_name, model_slug)
        prices = parse_prices(price_raw)
        return {
            "model_slug":   model_slug,
            "year":         year,
            "trim_name":    trim_name.strip(),
            **prices,
            "price_raw":    price_raw.strip(),
            "currency":     "AED",
            "url":          url,
            "scraped_at":   now,
        }

    def normalize_lines(text: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", ln).strip()
            for ln in text.splitlines()
            if re.sub(r"\s+", " ", ln).strip()
        ]

    def price_like(text: str) -> bool:
        if not _AED_RE.search(text):
            return False
        if re.search(r"\b(starting|calculator|monthly|similar|faq|highest|base price)\b", text, re.I):
            return False
        return bool(parse_prices(text)["price_avg_aed"])

    def add_unique(row_list: list[dict], trim_name: str, price_raw: str) -> None:
        trim_name = clean_trim_name(trim_name, model_slug)
        if not is_valid_trim(trim_name) or not price_like(price_raw):
            return
        row = make_row(trim_name, price_raw)
        key = (row["trim_name"].lower(), row["price_raw"])
        existing = {
            (r["trim_name"].lower(), r["price_raw"])
            for r in row_list
        }
        if key not in existing:
            row_list.append(row)

    def parse_trim_price_lines(lines: list[str]) -> list[dict]:
        parsed = []
        i = 0
        while i < len(lines):
            line = lines[i]

            if i + 1 < len(lines) and is_valid_trim(line) and price_like(lines[i + 1]):
                add_unique(parsed, line, lines[i + 1])
                i += 2
                continue

            if price_like(line):
                m = _AED_RE.search(line)
                if m:
                    trim = line[:m.start()].strip(" :-")
                    price = line[m.start():].strip()
                    add_unique(parsed, trim, price)
            i += 1
        return parsed

    def section_after_heading(*heading_patterns: str) -> list[str]:
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        for heading in headings:
            heading_text = heading.get_text(" ", strip=True)
            if not any(re.search(pattern, heading_text, re.I) for pattern in heading_patterns):
                continue

            section_lines = []
            for sib in heading.find_next_siblings():
                if not isinstance(sib, Tag):
                    continue
                if sib.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    break
                text = sib.get_text("\n", strip=True)
                if text:
                    section_lines.extend(normalize_lines(text))
            return section_lines
        return []

    # S0: DriveArabia's real trim data is under "Trim Prices" or
    # "Original Trim Prices". Keep this scoped before broad AED scans so
    # unrelated prices do not leak in.
    section_rows = parse_trim_price_lines(
        section_after_heading(r"^(?:original\s+)?trim\s+prices$")
    )
    if section_rows:
        log.debug("S0 matched: %d rows", len(section_rows))
        return section_rows

    log.debug("No scoped Trim Prices section parsed for %s %s", model_slug, year)
    return []

    # â”€â”€ S1: standard <table> with â‰¥2 <td> per row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                trim  = tds[0].get_text(strip=True)
                price = tds[1].get_text(strip=True)
                # also try last cell if middle cells exist
                if len(tds) > 2 and not _AED_RE.search(price):
                    price = tds[-1].get_text(strip=True)
                if trim and (_AED_RE.search(price) or re.search(r"\d{5,}", price)):
                    if is_valid_trim(trim):
                        rows.append(make_row(trim, price))
                    else:
                        log.debug("S1 rejected trim: %r", trim)
    if rows:
        log.debug("S1 matched: %d rows", len(rows))
        return rows

    # â”€â”€ S2: <tr> where one <td> has AED and the prev sibling <td> has trim â”€â”€â”€
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        for i, td in enumerate(tds):
            cell_text = td.get_text(strip=True)
            if _AED_RE.search(cell_text):
                trim_td = tds[i - 1] if i > 0 else None
                if trim_td:
                    trim = trim_td.get_text(strip=True)
                    if is_valid_trim(trim):
                        rows.append(make_row(trim, cell_text))
    if rows:
        log.debug("S2 matched: %d rows", len(rows))
        return rows

    # â”€â”€ S3: sibling element pairs â€” element with trim text next to price el â”€â”€
    #   DriveArabia uses various div/li layouts; look for any element whose
    #   text is AED-containing, then search previous siblings for trim text.
    price_els = soup.find_all(
        lambda tag: tag.name in ("div", "span", "li", "p", "td", "dd")
        and _AED_RE.search(tag.get_text())
        and len(tag.get_text(strip=True)) < 80
    )
    for el in price_els:
        price_text = el.get_text(strip=True)
        if not parse_prices(price_text)["price_avg_aed"]:
            continue
        trim = ""
        # search siblings first
        for sib in el.find_previous_siblings(limit=5):
            if not isinstance(sib, Tag):
                continue
            candidate = sib.get_text(strip=True)
            if is_valid_trim(candidate):
                trim = candidate
                break
        # then try parent's previous siblings
        if not trim and el.parent:
            for sib in el.parent.find_previous_siblings(limit=3):
                if not isinstance(sib, Tag):
                    continue
                candidate = sib.get_text(strip=True)
                if is_valid_trim(candidate):
                    trim = candidate
                    break
        rows.append(make_row(trim or "Unknown", price_text))
    if rows:
        log.debug("S3 matched: %d rows", len(rows))
        return rows

    # â”€â”€ S4: scan all text nodes for AED, pull trim from nearest named ancestor
    for node in soup.find_all(string=_AED_RE):
        price_raw = node.strip()
        if len(price_raw) > 100:
            continue
        if not parse_prices(price_raw)["price_avg_aed"]:
            continue
        trim = ""
        # walk up the DOM tree looking for a sibling/cousin with trim text
        el = node.parent
        for _ in range(4):           # up to 4 levels up
            if el is None:
                break
            for sib in el.find_previous_siblings(limit=4):
                if not isinstance(sib, Tag):
                    continue
                candidate = sib.get_text(strip=True)
                if is_valid_trim(candidate):
                    trim = candidate
                    break
            if trim:
                break
            el = el.parent
        rows.append(make_row(trim or "Unknown", price_raw))
    if rows:
        log.debug("S4 matched: %d rows", len(rows))
        return rows

    # â”€â”€ S5: last resort â€” full page text line scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #   Split all visible text into lines; look for consecutive lines where
    #   one looks like a trim and the next looks like a price.
    lines = [
        ln.strip()
        for ln in soup.get_text(separator="\n").splitlines()
        if ln.strip()
    ]
    i = 0
    while i < len(lines) - 1:
        if _AED_RE.search(lines[i + 1]) and is_valid_trim(lines[i]):
            rows.append(make_row(lines[i], lines[i + 1]))
            i += 2
            continue
        if _AED_RE.search(lines[i]) and is_valid_trim(lines[i - 1] if i > 0 else ""):
            rows.append(make_row(lines[i - 1], lines[i]))
            i += 2
            continue
        i += 1
    if rows:
        log.debug("S5 matched: %d rows", len(rows))

    return rows


# â”€â”€â”€ Model discovery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

KNOWN_MODELS = [
    "toyota-4runner", "toyota-avalon", "toyota-avanza", "toyota-c-hr",
    "toyota-camry", "toyota-corolla", "toyota-corolla-cross", "toyota-fortuner",
    "toyota-granvia", "toyota-hiace", "toyota-hilux", "toyota-innova",
    "toyota-land-cruiser", "toyota-land-cruiser-70-series",
    "toyota-land-cruiser-prado", "toyota-prado", "toyota-rav4", "toyota-rush",
    "toyota-sequoia", "toyota-starlet", "toyota-tundra", "toyota-vios",
    "toyota-yaris", "toyota-yaris-cross",
]


def discover_models(session: requests.Session) -> list[str]:
    log.info("Discovering Toyota models from %s", TOYOTA_INDEX)
    soup = fetch(session, TOYOTA_INDEX)
    if not soup:
        log.error("Failed to fetch Toyota index page â€” falling back to KNOWN_MODELS")
        return KNOWN_MODELS

    slugs   = set()
    pattern = re.compile(r"^/carprices/uae/toyota/(toyota-[^/]+)/?$")
    for a in soup.find_all("a", href=True):
        m = pattern.match(a["href"])
        if m:
            slugs.add(m.group(1))

    if not slugs:
        log.warning("No slugs found via <a> â€” falling back to KNOWN_MODELS")
        return KNOWN_MODELS

    models = sorted(slugs)
    log.info("Discovered %d models", len(models))
    return models


# â”€â”€â”€ Scrape loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def scrape(
    models: list[str],
    years: list[int],
    delay: float,
    output_path: Path,
    debug_html: bool = False,
) -> None:
    session     = make_session()
    all_records = []
    total       = len(models) * len(years)
    done        = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        for year in years:
            url = f"{BASE_URL}/carprices/uae/toyota/{model}/{year}/"
            done += 1
            log.info("[%d/%d] %s", done, total, url)

            soup = fetch(session, url)
            if soup is None:
                log.debug("Skipping %s %s (no response)", model, year)
                time.sleep(delay * 0.5)
                continue

            records = parse_trims(soup, model_slug=model, year=year,
                                  url=url, debug_html=debug_html)

            # defence-in-depth: drop rows where a competitor make leaked in
            before  = len(records)
            records = [r for r in records if not _KNOWN_MAKES.search(r["trim_name"])]
            if before - len(records):
                log.warning("  â†’ Dropped %d row(s) (competitor make in trim)", before - len(records))

            if records:
                log.info("  â†’ %d trims", len(records))
                all_records.extend(records)
            else:
                log.warning("  â†’ 0 trims parsed â€” try --debug-html to inspect page structure")

            time.sleep(delay)

    if not all_records:
        log.error("No records scraped. Not writing output file.")
        return

    fieldnames = [
        "model_slug", "year", "trim_name",
        "price_min_aed", "price_max_aed", "price_avg_aed",
        "price_raw", "currency", "url", "scraped_at",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    log.info("Wrote %d records â†’ %s", len(all_records), output_path)


# â”€â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main(
    start_year: int            = DEFAULT_START_YEAR,
    end_year:   int            = DEFAULT_END_YEAR,
    models:     list[str]|None = None,
    delay:      float          = DEFAULT_DELAY,
    output:     str            = "data/drivearabia_toyota_prices.csv",
    debug:      bool           = False,
    debug_html: bool           = False,
) -> None:
    """
    Jupyter usage:
        from scrape_drivearabia_toyota import main
        main(start_year=2020, end_year=2026, delay=1.0)

        # To inspect raw page structure for one model:
        main(start_year=2024, end_year=2024, models=["toyota-camry"], debug_html=True)
    """
    if debug or debug_html:
        logging.getLogger().setLevel(logging.DEBUG)

    if start_year > end_year:
        log.error("start_year (%d) must be <= end_year (%d)", start_year, end_year)
        return

    years   = list(range(start_year, end_year + 1))
    session = make_session()

    if models:
        resolved = models
        log.info("Using %d user-specified models", len(resolved))
    else:
        resolved = discover_models(session)
        if not resolved:
            log.error("Model discovery failed. Aborting.")
            return

    log.info("Year range: %dâ€“%d  |  Models: %d  |  Total pages: %d",
             start_year, end_year, len(resolved), len(resolved) * len(years))

    scrape(
        models=resolved,
        years=years,
        delay=delay,
        output_path=Path(output),
        debug_html=debug_html,
    )


# â”€â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Toyota vehicle prices from DriveArabia UAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                        help="First model year to scrape")
    parser.add_argument("--end-year",   type=int, default=DEFAULT_END_YEAR,
                        help="Last model year to scrape (inclusive)")
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Specific model slugs (default: auto-discover)")
    parser.add_argument("--delay",      type=float, default=DEFAULT_DELAY,
                        help="Seconds between requests")
    parser.add_argument("--output",     default="data/drivearabia_toyota_prices.csv",
                        help="Output CSV path")
    parser.add_argument("--debug",      action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--debug-html", action="store_true",
                        help="Dump page structure to stdout (use with --models + 1 year to isolate)")
    args, _ = parser.parse_known_args()

    main(
        start_year=args.start_year,
        end_year=args.end_year,
        models=args.models,
        delay=args.delay,
        output=args.output,
        debug=args.debug,
        debug_html=args.debug_html,
    )


if __name__ == "__main__":
    _cli_main()


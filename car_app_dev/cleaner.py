#!/usr/bin/env python3
"""
vehicle_model_cleaner_v2.py  —  VIN Dataset Model-Column Cleaner  (v2)
=======================================================================
An enhanced, make-aware cleaning pipeline with 10 steps:

  0. Arabic / Farsi transliteration  (new: translate before classify)
  1. Classify & discard junk         (years, Arabic, numbers, generics)
  2. Normalize whitespace & case
  3. Make-aware wrong-brand removal  (Yamaha model listed under Honda, etc.)
  4. Correct known misspellings       (hand-curated per-make & global map)
  5. Strip brand prefixes             ("HONDA CBF160" → "CBF 160")
  6. Strip body-type suffixes         ("COROLLA/SEDAN" → "COROLLA")
  7. Remove model-year tags           ("CBF16026MY" → "CBF 160")
  8. Remove embedded calendar years  ("COROLLA 2019" → "COROLLA")
  9. Remove standalone CC specs       ("PULSAR 220 CC" → "PULSAR 220")

Usage
-----
    python vehicle_model_cleaner_v2.py --input raw.csv --col Model --make Make --output cleaned.csv

    # or call from your own code:
    from vehicle_model_cleaner_v2 import process_model, process_dataframe

Outputs
-------
    cleaned.csv        — original file + model_clean, model_status, model_changes
    model_mapping.csv  — unique (make, original_model) → cleaned lookup for QA
"""

from __future__ import annotations
import re, sys, argparse
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — ARABIC / FARSI TRANSLITERATION TABLE
#
#  WHY: Many entries are Arabic / Farsi transliterations of English model names.
#  Translating them before the rest of the pipeline lets every downstream step
#  work uniformly on ASCII-ish text.  The table is intentionally complete for
#  every Arabic entry found in this dataset, grouped by make for traceability.
#
#  Translation methodology:
#    • Phonetic Arabic → closest canonical English model name
#    • Generic Arabic phrases (دراجة نارية = "motorcycle") → JUNK key
#    • Pick-up / body-type phrases → body-type JUNK keys
#    • When the Arabic is a brand name only (مولدا = Honda) → '' (JUNK)
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_TRANSLATIONS: dict[str, str] = {

    # ── BAJAJ ────────────────────────────────────────────────────────────────
    'اف اكس 150 الترا':  'FX 150 ULTRA',      # "FX 150 Ultra"
    'افينجر':            'AVENGER',             # "Avenger"
    'بولسار':            'PULSAR',              # "Pulsar"
    'دراجة نارية':       'MOTORCYCLE',          # "motorcycle" → junk
    'دراجة ناريه':       'MOTORCYCLE',          # variant spelling
    'سي 150':            'C 150',               # "C 150"
    'سي تي 100':         'CT 100',              # "CT 100"
    'مي سي 150':         'CT 150',              # "CT 150" (phonetic)
    'ميتي 100':          'CT 100',              # "CT 100" (phonetic)

    # ── HONDA ─────────────────────────────────────────────────────────────────
    'اف 150 سي بي':             'CBF 150',          # "AF 150 CB" → CBF 150
    'بي اف 150':                'BF 150',            # "BF 150"
    'بی اف 150':                'BF 150',            # Farsi variant
    'تي ان تي 150':              'TNT 150',           # "TNT 150" (Benelli under Honda?)
    'جي ال 1800':               'GL 1800',           # "GL 1800"
    'دراجة توصيل الباي':         'DELIVERY MOTORCYCLE',
    'دراجة توصيل طلبات':         'DELIVERY MOTORCYCLE',
    'سي بي اف 150':              'CBF 150',
    'سي بي اف 160 يـ':           'CBF 160',
    'سي بي اف 160 يونيكورن':     'CBF 160 UNICORN',
    'سي جي 125':                 'CG 125',
    'سی بی اف 150':              'CBF 150',          # Farsi variant
    'سی بی اف 160':              'CBF 160',
    'سی بی اف 160 ی':            'CBF 160',
    'شادو':                      'SHADOW',
    'كوندا جي ال 1800':          'GL 1800',          # "Honda GL 1800" phonetic
    'مولدا':                     '',                  # garbled brand-only → junk
    'نيكورن 160':                'UNICORN 160',
    'يوليكورن 160':              'UNICORN 160',
    'يونيكورت 160':              'UNICORN 160',      # "unicourt" typo
    'يونيكورن':                  'UNICORN',
    'يونيكورن 160':              'UNICORN 160',
    'يونيكورن 180':              'UNICORN 160',      # 180 doesn't exist → 160
    'يونيكورن-160':              'UNICORN 160',
    'يونيكون 160':               'UNICORN 160',
    'يونيكونت 160':              'UNICORN 160',
    'پولیگورن':                  'UNICORN',           # Farsi phonetic
    'یونیکورن':                  'UNICORN',           # Farsi
    'سي بي اف 160 يـ':           'CBF 160',

    # ── KIA ──────────────────────────────────────────────────────────────────
    'بيك اب شحن':               'PICKUP',            # "cargo pickup" → junk
    'بيك اب كابينة واحدة':      'SINGLE CABIN PICKUP',
    'يونجو':                    'BONGO',             # "Yongo" = KIA Bongo

    # ── NISSAN ───────────────────────────────────────────────────────────────
    # 'بيك اب شحن' already above
    'بيك اب كابينه مزدوجه':     'DOUBLE CABIN PICKUP',
    'سيفليان':                  'CIVILIAN',          # Nissan Civilian bus
    'صني':                      'SUNNY',
    'صني 1.3':                  'SUNNY 1.3',
    'صني 1.4':                  'SUNNY 1.4',
    'صني 1.5':                  'SUNNY 1.5',
    'صني 1.6':                  'SUNNY 1.6',

    # ── TOYOTA ───────────────────────────────────────────────────────────────
    'استیشن':                   'STATION',           # body-type → junk
    'افنزا':                    'AVANZA',
    'برادو':                    'PRADO',
    'بريفيا':                   'PREVIA',
    'تاكوما':                   'TACOMA',
    'حافلة نقل خاص':            'PRIVATE BUS',       # "private transport bus"
    'راف ٤':                    'RAV4',               # ٤ = Arabic 4
    'كامرى':                    'CAMRY',
    'كامري':                    'CAMRY',
    'كورولا':                   'COROLLA',
    'كورولا 1.3':               'COROLLA 1.3',
    'كورولا 1.8':               'COROLLA 1.8',
    'لاند كروزر':               'LAND CRUISER',
    'هاي لوكس':                 'HILUX',
    'هايلوكس':                  'HILUX',

    # ── CHANGAN ──────────────────────────────────────────────────────────────
    'سي اس 35':                 'CS35',

    # ── FOTON ────────────────────────────────────────────────────────────────
    'سي 2 اس دبلي':             'VIEW C2S W',

    # ── CMC ──────────────────────────────────────────────────────────────────
    'فان شحن':                  'VAN CARGO',

    # ── GWM / HAVAL ──────────────────────────────────────────────────────────
    'وينجل':                    'WINGLE',

    # ── GEELY ────────────────────────────────────────────────────────────────
    'ني ام جراند':              'EM GRAND',

    # ── ISUZU ────────────────────────────────────────────────────────────────
    'بيك آب شحن':               'PICKUP',
    'بيك اب کابینه مزدوجه':    'DOUBLE CABIN PICKUP',

    # ── MERCEDES BENZ ────────────────────────────────────────────────────────
    'اتيجو 1725':               'ATEGO 1725',
    'ال اس 350':                'LS 350',
    'ام ال 350':                'ML 350',
    'في 230':                   'V 230',
    'في 250':                   'V 250',

    # ── LEXUS ────────────────────────────────────────────────────────────────
    'ال اس 430':                'LS 430',
    'جي اس 430':                'GS 430',
    'لي اس 300':                'LS 300',

    # ── BMW ──────────────────────────────────────────────────────────────────
    'آي 330':                   'I 330',             # likely 330i
    'اف 650 جي اس':             'F 650 GS',

    # ── SUZUKI ───────────────────────────────────────────────────────────────
    'جي اس اكس ار 1000':        'GSX-R1000',

    # ── HYUNDAI ──────────────────────────────────────────────────────────────
    'اتش دي 65':                'HD 65',
    'ايفنت':                    'AVANTE',

    # ── MITSUBISHI ───────────────────────────────────────────────────────────
    # 'بيك اب شحن' already above
    # 'بيك اب کابینه مزدوجه' already above

    # ── PIAGGIO / VESPA ──────────────────────────────────────────────────────
    'اس اكس ال 150':            'SXL 150',
    'اس واي 15 تي يو':          'SY 150 TU',

    # ── SANYA ────────────────────────────────────────────────────────────────
    'اس واي 150 تي':            'SY 150 T',
    'اس ولی 150 تے':            'SY 150 T',         # Urdu variant

    # ── VICTORY ──────────────────────────────────────────────────────────────
    'سيكتيس 175':               'SIXTIES 175',

    # ── TVS ──────────────────────────────────────────────────────────────────
    'اباشي ار تي ار 20':        'APACHE RTR 200',
    'ستار اتش ال اكس في اس 1 تي في اس': 'STAR HLX 150',

    # ── HERO ─────────────────────────────────────────────────────────────────
    'دراجة توصيل طلبات':         'DELIVERY MOTORCYCLE',
    'دراجة ناريه':               'MOTORCYCLE',

    # ── SYM ──────────────────────────────────────────────────────────────────
    'می سی 125':                'MSC 125',

    # ── NISSAN (additional) ──────────────────────────────────────────────────
    'quest':                    'QUEST',             # lowercase leak
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — JUNK DETECTION TABLES
# ══════════════════════════════════════════════════════════════════════════════

JUNK_SET: set[str] = {
    # empty / placeholder
    '', '-', '--', '---', 'N/A', 'NA', '0', 'NONE', 'NIL', 'NUN', 'NOR',
    'OTHER', 'OTHERS', 'UNKNOWN', 'UNIVERSAL',
    # generic vehicle-type words (carry zero model information)
    'CAR', 'BUS', 'TRUCK', 'VAN', 'CYCLE', 'BIKE', 'SCOOTER',
    'MOTORCYCLE', 'MOTOR CYCLE', 'MOTOR BIKE', 'MOTORBIKE',
    'DELIVERY BIKE', 'DELIVERY MOTORCYCLE', 'ELECTRIC BIKE',
    'DIESEL', 'TURBO', 'TRAILER', 'TRAILOR', 'DUMP TRUCK',
    'SCHOOL BUS', 'MOTORHOME', 'LIGHT TRUCK', 'LIGHT VEHICLE',
    'PANEL', 'VAN CARGO', 'CARGO VAN', 'CARGO', 'STATION',
    'SEDAN', 'COUPE', 'HATCHBACK', 'SUV', 'PICKUP', 'PICK UP',
    'SINGLE CABIN', 'DOUBLE CABIN', 'PICKUP CARGO',
    'SINGLE CABIN PICKUP', 'DOUBLE CABIN PICKUP',
    'WATER TANK', 'TRANSPORT', 'TRANSPORTE', 'FLAT WOOD',
    'FLAT WOOD SEDAN', 'PRIVATE BUS',
    # real-world data anomalies
    'JUSTIN BIEBER', 'GALAXY M13 5G', 'HONOR X9C',
    'DELIVERY MOTORCYCLE', 'MOTOR CYCLE',
}

ARABIC_RE          = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
PURE_DIGIT_RE      = re.compile(r'^[\d\s.,/]+$')     # "160", "0.60", "1000"
STANDALONE_YEAR_RE = re.compile(r'^(19[5-9]\d|20[0-3]\d)$')
DOLLAR_HASH_RE     = re.compile(r'^[\$#]')

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — WRONG-BRAND MODELS (make-aware)
#
#  WHY: Some entries are real vehicle models but belong to a DIFFERENT brand.
#  e.g. "HAYABUSA" listed under HONDA (it's Suzuki), "YZF R1" under HONDA
#  (it's Yamaha). We flag these as WRONG_BRAND so they can be investigated,
#  rather than silently corrupting the cleaned output.
#
#  Format:  make → set of model strings that definitely belong to other brands
# ══════════════════════════════════════════════════════════════════════════════

WRONG_BRAND_MODELS: dict[str, set[str]] = {
    'HONDA': {
        # Suzuki models
        'HAYABUSA', 'GSX-R1000', 'GSXR1000', 'GSX R1000', 'GSXR',
        'BOULEVARD', 'BOULEVARD 250', 'INTRUDER', 'BURGMAN',
        # Yamaha models
        'YZF R1', 'YZF R3', 'YZF R6', 'YZF R7', 'MT-09', 'MT-07', 'YZFR7',
        'TMAX', 'NMAX', 'R6',
        # Harley-Davidson
        'FLHCS', 'FLHTK', 'FLTRXSE', 'HERITAGE SOFTAIL',
        # Kawasaki
        'NINJA 300', 'ZX6R', 'ZX10R', 'VULCAN',
        # KTM
        'DUKE', 'ADVENTURE',
        # Piaggio / Vespa (often confused)
        'VESPA', 'VESPA 150', 'VESPA VXL', 'VESPA VXL 150', 'VXL 150',
        # Other brands
        'ROCKY',  # Daihatsu / Chery
        'SLINGSHOT',
    },
    'TOYOTA': {
        # Nissan
        'NIO ES8',
        # BMW
        'GL770ULTRA',
        # Vespa
        'VESPA VXL', 'VESPA VXL 150',
        # Yamaha
        'MT 03',
        # Generic body types (already handled by JUNK_SET but listed for clarity)
    },
    'NISSAN': {
        # Infiniti (technically parent but different WMI)
        'QX 56',
        # Kawasaki
        'SLING SHOT',
        # Honda
        'VESPA 150',
    },
    'BMW': {
        # Harley-Davidson
        'HARLEY DAVIDSON', 'HERITAGE SOFTAIL',
        # Honda
        'VESPA', 'VESPA VXL', 'VESPA VXL 150',
        # Kawasaki
        'SLINGSHOT', 'SLING SHOT', 'SLINGS',
        # Suzuki
        'SY150T', 'SCOMADI 2001',
        # Smart
        'SMART',
    },
    'BAJAJ': {
        # Honda / other brand models that leaked in
        'ACCORD', 'FLHTCUI',
        # Yamaha
        'YZF R1',
        # Samsung (phone)
        'GALAXY M13 5G',
    },
    'HONDA': {
        'ROCKY',  # Daihatsu
    },
    'KIA': {
        # GAC model
        'GAC - EMZOOM 1.5TG GS',
        # Yamaha
        'YAMAHA MTN 890',
    },
    'CHEVROLET': {
        # Kawasaki
        'NINJA 300',
    },
    'FORD': {
        # Lamborghini
        'HURACAN',
        # Chevrolet
        'CHALLENGER',
    },
    'LEXUS': {
        # Harley
        'HARLEY DAVIDSON FAT',
        # Honda
        'CH150T-34',
        # Nissan
        'NX 350 H',  # That's a Lexus actually, keep
    },
    'PIAGGIO': {
        # Aprilia (Piaggio group so keep, but flag)
        # Chieftain is Indian Motorcycle
        'CHIEFTAIN 1400',
        # Harley
        'HARLEY DAVIDSON',
    },
    'GEELY': {
        # Volvo (parent company, different WMI)
        # Keep Volvo models as they may share data, don't flag
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BRAND PREFIX TABLE
# ══════════════════════════════════════════════════════════════════════════════

# Sorted longest-first so "HARLEY DAVIDSON" matches before "HARLEY"
BRAND_PREFIXES: list[str] = sorted([
    'HARLEY DAVIDSON', 'HARLEY-DAVIDSON', 'ROYAL ENFIELD', 'MERCEDES BENZ',
    'MERCEDES-BENZ', 'INDIAN MOTORCYCLE',
    'HONDA', 'YAMAHA', 'SUZUKI', 'KAWASAKI', 'KTM', 'BAJAJ', 'BAJAJA',
    'MERCEDES', 'TOYOTA', 'NISSAN', 'HYUNDAI', 'KIA', 'FORD', 'CHEVROLET',
    'MITSUBISHI', 'ISUZU', 'FUSO', 'HINO', 'TVS', 'HERO', 'DUCATI',
    'TRIUMPH', 'PIAGGIO', 'VESPA', 'APRILIA', 'HARLEY', 'MG', 'JAC',
    'TATA', 'MAHINDRA', 'PHONDA', 'THONDA',
    'HONDA-', 'BMW',  # "BMW 320I" → "320I", keep
    'KAWASAKI', 'NISSAN', 'FORD', 'DODGE', 'MAZDA', 'SUBARU',
    'FENGON',   # DFSK brand prefix
    'SHARMAX',  # keep model part only
], key=len, reverse=True)

# Brands where stripping the prefix would leave nothing useful — skip stripping
BRAND_ONLY_ENTRIES: set[str] = {
    'BAJAJ', 'BAJAJA', 'BMW', 'HONDA', 'YAMAHA', 'SUZUKI', 'KAWASAKI',
    'KTM', 'TOYOTA', 'NISSAN', 'KIA', 'HYUNDAI', 'FORD', 'CHEVROLET',
    'MERCEDES', 'MITSUBISHI', 'ISUZU', 'TATA', 'MAHINDRA',
    'PHONDA', 'THONDA', 'HARLEY DAVIDSON', 'HARLEY',
    'DUCATI', 'TRIUMPH', 'PIAGGIO', 'VESPA', 'APRILIA',
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BODY-TYPE SUFFIX PATTERN
# ══════════════════════════════════════════════════════════════════════════════

BODY_SUFFIX_RE = re.compile(
    r'\s*[/\(]\s*('
    r'PICK\s*UP|PICKUP|SINGLE[\s-]*CABIN\s*PICK\s*UP|DOUBLE[\s-]*CABIN\s*PICK\s*UP|'
    r'SINGLE[\s-]*CABIN|DOUBLE[\s-]*CABIN|'
    r'STATION\s*WAGON|STATION|SEDAN|COUPE|HATCHBACK|VAN\s*CARGO|'
    r'BOX\s*CARRIER\s*TRUCK|BOX\s*CARRIER|BOX|TRUCK|BUS|COACH|CHASSIS|'
    r'CARGO|PANEL\s*VAN|MOTOR\s*HOME|MOTORHOME|'
    r'FOODSTUFF\s*FRIDGE\s*VAN|MOBILE\s*WORKSHOP|'
    r'DRINKING\s*WATER\s*TANK|MOTORCYCLE|ELECTRIC\s*BIKE|ELECTRIC|'
    r'STD\s*EX\s*PICKUP|SCHOOL\s*BUS|VAN\s*PASSENGER|VAN\s*CARGO|'
    r'PICK\s*UP\s*DOUBLE\s*CABIN|PICK\s*UP\s*CARGO|FOODSTUFF)'
    r'.*$',
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — INLINE PATTERN STRIPS
# ══════════════════════════════════════════════════════════════════════════════

YEAR_RE   = re.compile(r'\b(19[5-9]\d|20[0-3]\d)\b')
MY_TAG_RE = re.compile(r'\s+\d{2}(MY|Y)\b', re.IGNORECASE)   # "26MY", "25Y"
CC_RE     = re.compile(r'(?<=\d)\s*CC\b|\s+CC\b', re.IGNORECASE)

# Additional inline patterns
_4WD_RE      = re.compile(r'\s*\(?(4WD|2WD|AWD|RWD|FWD)\)?\s*$', re.IGNORECASE)
_TRAILING_JUNK_RE = re.compile(
    r'\s+(PICK\s*UP|SINGLE.CABIN|DOUBLE.CABIN|CARGO|STATION|SEDAN|COUPE|VAN)$',
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — GLOBAL CORRECTIONS MAP  (make-independent)
# ══════════════════════════════════════════════════════════════════════════════

CORRECTIONS: dict[str, str] = {

    # ── UNICORN / Honda CBF160 ────────────────────────────────────────────────
    'UNICRN':                   'UNICORN',
    'UNICRN 160':               'UNICORN 160',
    'UNICCRN':                  'UNICORN',
    'UNICCRN 160':              'UNICORN 160',
    'UNICRON':                  'UNICORN',
    'UNICRON 160':              'UNICORN 160',
    'UNIKORN':                  'UNICORN',
    'UNIKORN 160':              'UNICORN 160',
    'UNICON':                   'UNICORN',
    'UNICON ':                  'UNICORN',
    'UNICON 160':               'UNICORN 160',
    'UNICORK 160':              'UNICORN 160',
    'UNICO':                    'UNICORN',
    'UN CORN':                  'UNICORN',
    'UN.CORN':                  'UNICORN',
    'UNIRON':                   'UNICORN',
    'UNIRON 160':               'UNICORN 160',
    'UNIC ORN':                 'UNICORN',
    'UNI CORN':                 'UNICORN',
    'UNI CORN 160':             'UNICORN 160',
    'UNICORN 16':               'UNICORN 160',
    'UNICORN 100':              'UNICORN',
    'UNICORN 10':               'UNICORN',
    'UNICORN 166':              'UNICORN 160',
    'UNICORN 167':              'UNICORN 160',
    'UNICORN 180':              'UNICORN 160',   # 180 variant doesn't exist
    'UNICRN 16':                'UNICORN 160',
    'YULICORN':                 'UNICORN',
    'YULICORN 160':             'UNICORN 160',
    'YULICORN 180':             'UNICORN 160',
    'YUNICORN':                 'UNICORN',
    'YUNICORN 160':             'UNICORN 160',
    'YUNICORN 180':             'UNICORN 160',
    'YULIKORN':                 'UNICORN',
    'YULIKORN 160':             'UNICORN 160',
    'YULICORN160':              'UNICORN 160',
    'MONDA UNICORN':            'UNICORN',
    'BUNICORN':                 'UNICORN',
    'JNICORN':                  'UNICORN',
    'JNICORN 160':              'UNICORN 160',
    'PUNICORN':                 'UNICORN',
    'MUNICORN':                 'UNICORN',
    'MUNICORN 160':             'UNICORN 160',
    'UNICORN P160':             'UNICORN 160',
    'UNICORNP160':              'UNICORN 160',
    'UNIK':                     'UNICORN',
    'UNICORNS!':                'UNICORN',
    'UNICORNT':                 'UNICORN',
    'UNICORNS':                 'UNICORN',
    'YUNIKORN':                 'UNICORN',
    'YUNIKORN 160':             'UNICORN 160',
    'YUNIKORN160':              'UNICORN 160',
    'YULICORN 160':             'UNICORN 160',
    'YUNIKORN 160':             'UNICORN 160',
    'YUMSUN':                   'UNICORN',          # phonetic Arabic transliteration
    'YONGO':                    'BONGO',             # KIA Bongo (not Unicorn)
    'UNI160BSVI':               'UNICORN 160',
    'UNICORN CBF':              'CBF 160 UNICORN',
    'UNICORN MOTORCYCLE':       'CBF 160 UNICORN',
    'UNICORN MOTORCYCLE DELIVERY': 'CBF 160 UNICORN',
    'CBF 160 UNICORN PC':       'CBF 160 UNICORN',
    'AF 160 UNICORN PC':        'CBF 160 UNICORN',
    'AF 160 UNICORN':           'CBF 160 UNICORN',
    'HONDA UNICORN 150':        'CBF 150 UNICORN',
    'HONDA UNICORN 160':        'CBF 160 UNICORN',
    'HONDA UNICORN':            'UNICORN',
    'MONDA UNICORN':            'UNICORN',
    'UINCORN 160':              'UNICORN 160',
    'UNIRON 160':               'UNICORN 160',
    'UNIQUE 160':               'UNICORN 160',
    'BOMBKORN 160':             'UNICORN 160',
    'CINGIN':                   'UNICORN',           # phonetic
    'JNICORN':                  'UNICORN',
    'MODCI':                    'UNICORN',           # garbled
    'PONDA ACTIVA 125 21D':     'ACTIVA 125',        # "Honda" → "Ponda"
    'YCBF 160':                 'CBF 160',
    'YCBF160':                  'CBF 160',
    'YULICORN160':              'UNICORN 160',
    'B105':                     'B 105',
    'B1025':                    'B 1025',

    # ── CBF series ───────────────────────────────────────────────────────────
    'CBF 16025MY':              'CBF 160',
    'CBF 16025Y':               'CBF 160',
    'CBF 16026MY':              'CBF 160',
    'CBF 16026Y':               'CBF 160',
    'CBF1606MY':                'CBF 160',
    'CBF16026':                 'CBF 160',
    'CBF16026MY':               'CBF 160',
    'CBF16026Y':                'CBF 160',
    'CBF-160':                  'CBF 160',
    'CBF160F':                  'CBF 160F',
    'CBF1505':                  'CBF 150',
    'CBF150MA':                 'CBF 150 MA',
    'CBF150MA UNICORN':         'CBF 150 MA',
    'CBF150MA UNICORN (MOTORCYCLE)': 'CBF 150 MA',
    'AF 160 CB':                'CBF 160',
    'AF 150 CB':                'CBF 150',
    'CBF160 UNICORN':           'CBF 160 UNICORN',
    'CBF 160 UNICORN':          'CBF 160 UNICORN',
    'CEF150':                   'CBF 150',
    'CBF 16026':                'CBF 160',
    'CBF 16026 MY':             'CBF 160',
    'CBF 160 26MY':             'CBF 160',
    'CBF160 26MY':              'CBF 160',

    # ── PULSAR (Bajaj) ────────────────────────────────────────────────────────
    'POLSAR':                   'PULSAR',
    'BULSAR':                   'PULSAR',
    'PULS R':                   'PULSAR',
    'PUL.SAP. 150':             'PULSAR 150',
    'PULS R 220CC':             'PULSAR 220',
    'PULSAR 1800C':             'PULSAR 180',        # typo: 1800C → 180
    'PULSAR 180C':              'PULSAR 180',
    'PULSAR 200C':              'PULSAR 200',
    'PULSAR 220C':              'PULSAR 220',
    'PULSAR 80CC':              'PULSAR',            # 80CC Pulsar doesn't exist
    'TV BAJAJ 220 CC':          'PULSAR 220',

    # ── APACHE (TVS) ─────────────────────────────────────────────────────────
    'APPACHE':                  'APACHE',
    'APPACHE 180':              'APACHE RTR 180',
    'APACHE RTR160':            'APACHE RTR 160',
    'APACHE RTR180':            'APACHE RTR 180',
    'APACHE RTR200':            'APACHE RTR 200',
    'APACHE RTR160 4V':         'APACHE RTR 160 4V',
    'APACHE RTR 160 4 TD FD SL': 'APACHE RTR 160 4V',

    # ── HAYABUSA (Suzuki GSX1300R) ────────────────────────────────────────────
    'HAYA BUSA':                'HAYABUSA',
    'HAYABUSA 1300R':           'GSX1300R HAYABUSA',
    'HAUA BUSE':                'HAYABUSA',
    'GSX 1300 R':               'GSX1300R',
    'HAUA BUSE/MOTORCYCLE':     'HAYABUSA',

    # ── GSX-R normalization (Suzuki) ─────────────────────────────────────────
    'GSXR':                     'GSX-R',
    'GSX R':                    'GSX-R',
    'GSXR1000':                 'GSX-R1000',
    'GSXR750':                  'GSX-R750',
    'GSXR600':                  'GSX-R600',
    'GSXR1300':                 'GSX-R1300',
    'GSXR 1000':                'GSX-R1000',
    'GSXR 750':                 'GSX-R750',
    'GSXR 600':                 'GSX-R600',
    'GSX-R 1':                  'GSX-R1000',
    'GSX-R 100':                'GSX-R1000',
    'GSX-R MOTORCYCLE':         'GSX-R',
    'GSXR10004':                'GSX-R1000',
    'GSXR1000 AL6':             'GSX-R1000',
    'GSXR1000R':                'GSX-R1000R',
    'GSXS1000A':                'GSX-S1000',
    'GSXS 1000':                'GSX-S1000',
    'GSXS 1000 YA':             'GSX-S1000',
    'GSXS 1000 YAL 9':          'GSX-S1000',
    'GSXS1000AL8':              'GSX-S1000',
    'GSXS1000TRQ':              'GSX-S1000',

    # ── BROUGHAM (Cadillac / Lincoln) ─────────────────────────────────────────
    'BROYGAM':                  'BROUGHAM',
    'BROYGHAM':                 'BROUGHAM',
    'BROGHAM':                  'BROUGHAM',
    'BROUGHTAM':                'BROUGHAM',
    'BROGH':                    'BROUGHAM',
    'BROUGH':                   'BROUGHAM',
    'BROYGAN':                  'BROUGHAM',
    'CADILLAC BROUGH':          'BROUGHAM',
    'CADILLAC CEVEL':           'SEVILLE',
    'CEVEL':                    'SEVILLE',

    # ── SLINGSHOT ─────────────────────────────────────────────────────────────
    'SLING SHOT':               'SLINGSHOT',
    'SLINGS':                   'SLINGSHOT',

    # ── Smart FORTWO ─────────────────────────────────────────────────────────
    'FORT-WO':                  'FORTWO',

    # ── LAND CRUISER ─────────────────────────────────────────────────────────
    'LAND CRUISE':              'LAND CRUISER',
    'LAND CRUISE RA':           'LAND CRUISER',
    'HAI LUX':                  'HILUX',
    'HAIULUX':                  'HILUX',

    # ── DISCOVERY (Land Rover) ────────────────────────────────────────────────
    'DISCO':                    'DISCOVERY',
    'DEVENDER':                 'DEFENDER',
    'DEVENDR':                  'DEFENDER',

    # ── GOLD WING (Honda) ─────────────────────────────────────────────────────
    'COLD WING':                'GOLD WING',
    'GOLDWING':                 'GOLD WING',
    'GOLDWING DCT':             'GOLD WING DCT',
    'GOLDWING TOUR':            'GOLD WING TOUR',
    'GOLDWING TOUR AUTO':       'GOLD WING TOUR',
    'GL 1800':                  'GOLD WING GL1800',
    'GL1800 D':                 'GOLD WING GL1800',
    'GL1800A':                  'GOLD WING GL1800',
    'GL1800XS 2GS':             'GOLD WING GL1800',
    'HONDA GL 1800':            'GOLD WING GL1800',
    'HONDA GOLD WING':          'GOLD WING',
    'GL 1806/MOTORCYCLE':       'GOLD WING GL1800',
    'GL1806 (MOTORCYCLE)':      'GOLD WING GL1800',

    # ── INTRUDER (Suzuki) ─────────────────────────────────────────────────────
    'IN TRUDER':                'INTRUDER',

    # ── ELECTRA GLIDE (Harley) ────────────────────────────────────────────────
    'ELECTRAGLIDE':             'ELECTRA GLIDE',

    # ── DOMINAR (Bajaj) ──────────────────────────────────────────────────────
    'DOMINOR':                  'DOMINAR',
    'DOMINOR 400CC':            'DOMINAR 400',
    'DOMINAR 400CC':            'DOMINAR 400',

    # ── GRAND VITARA (Suzuki) ────────────────────────────────────────────────
    'GRAND VITA':               'GRAND VITARA',

    # ── AVENGER ──────────────────────────────────────────────────────────────
    'AFVENGER':                 'AVENGER',
    'AVEGER':                   'AVENGER',
    'AVANTY':                   'AVANTI',

    # ── FORERUNNER vs 4RUNNER ─────────────────────────────────────────────────
    'FORERUNNER':               '4RUNNER',

    # ── CHEROKEE ─────────────────────────────────────────────────────────────
    'CHERO':                    'CHEROKEE',

    # ── NINJA (Kawasaki) ─────────────────────────────────────────────────────
    'NINED POE':                'NINJA',

    # ── RAV4 (Toyota) ────────────────────────────────────────────────────────
    'RAFF 4':                   'RAV4',
    'RAV 4':                    'RAV4',
    'RAIZE':                    'RAIZE',

    # ── SILVERADO ────────────────────────────────────────────────────────────
    'SELVERADO':                'SILVERADO',

    # ── TUNDRA ───────────────────────────────────────────────────────────────
    'TANDRA':                   'TUNDRA',
    'TANDRA PICK UP DOUBLE CABIN': 'TUNDRA',

    # ── NIAS / NINJA ─────────────────────────────────────────────────────────
    'NIAS 350':                 'NINJA',

    # ── HIACE (Toyota) ───────────────────────────────────────────────────────
    'HI ACE':                   'HIACE',

    # ── Alfa Romeo duplicates ─────────────────────────────────────────────────
    'ALFA ROMEO 4C':            '4C',
    'GIULIA':                   'GIULIA',   # normalize case duplicate

    # ── Bentley duplicates ────────────────────────────────────────────────────
    'BENTLEY GT':               'CONTINENTAL GT',

    # ── Aston Martin duplicates ───────────────────────────────────────────────
    'DBX 707':                  'DBX707',
    'RAPIDE S':                 'RAPIDE S',

    # ── Audi RS normalization ─────────────────────────────────────────────────
    'RSQ 8':                    'RSQ8',
    'AUDI RS Q8':               'RS Q8',

    # ── BMW normalization ─────────────────────────────────────────────────────
    'BMW AR NAET':              'I3',        # iQ/iNext-related garble
    'AR NAET':                  'I3',
    'SLINGSHOT SLINGS':         'SLINGSHOT',

    # ── SHARMAX / corrections ────────────────────────────────────────────────
    'SHARMAX RR 801V U':        'SHARMAX RR 801',
    'SHARMAX/RR 661':           'SHARMAX RR 661',
    'SHARMAX ENDURO2':          'SHARMAX ENDURO 2',

    # ── KROSS / CROSS ─────────────────────────────────────────────────────────
    'KROSS VICTORY TOUR':       'CROSS COUNTRY TOUR',

    # ── GRANDEUR (Hyundai) ────────────────────────────────────────────────────
    'GRANDEU':                  'GRANDEUR',
    'GRANDYUOR':                'GRANDEUR',
    'GRANDYUOR / SEDAN':        'GRANDEUR',
    'GRAND YUOR':               'GRANDEUR',
    'GRAND YUOR / SEDAN':       'GRANDEUR',
    'GRAND YUR':                'GRANDEUR',
    'GENES G90':                'GENESIS G90',

    # ── SUNNY (Nissan) ────────────────────────────────────────────────────────
    'SUNNY/SEDAN':              'SUNNY',
    'NISSAN SUNNY 1.3':         'SUNNY 1.3',
    'NISSAN SUNNY 1.6':         'SUNNY 1.6',

    # ── PATROL (Nissan) ──────────────────────────────────────────────────────
    'PATROL SUPER SA':          'PATROL SUPER SAFARI',
    'NISSAN / PATROL / STATION': 'PATROL',
    'PATROLYYYY':               'PATROL',
    'PATROL GL V':              'PATROL GLV',

    # ── COROLLA (Toyota) ─────────────────────────────────────────────────────
    'COROLLA LI':               'COROLLA',
    'TOYOTA COROLLA 2.0':       'COROLLA',

    # ── CAMRY (Toyota) ───────────────────────────────────────────────────────
    'TOYOTA CAMRY':             'CAMRY',

    # ── ALTIMA (Nissan) ──────────────────────────────────────────────────────
    'ALTIAMA':                  'ALTIMA',

    # ── FORTUNER (Toyota) ────────────────────────────────────────────────────
    # NOTE: 4RUNNER ≠ FORTUNER — they are different vehicles.
    # "FORERUNNER" (typo) → 4RUNNER (correct), NOT Fortuner.

    # ── PRADO (Toyota) ───────────────────────────────────────────────────────
    'PRADO GX':                 'LAND CRUISER PRADO',

    # ── MURANO (Nissan) ──────────────────────────────────────────────────────
    'MORANOA':                  'MURANO',
    'MUNARO':                   'MURANO',

    # ── QASHQAI (Nissan) ─────────────────────────────────────────────────────
    'ROGIE':                    'ROGUE',
    'ROUGE':                    'ROGUE',

    # ── HILUX (Toyota) ───────────────────────────────────────────────────────
    'HILUX RE':                 'HILUX',
    'HILUX ADVENTURE':          'HILUX',

    # ── VESPA normalization ───────────────────────────────────────────────────
    'VESPA VXL 150':            'VESPA VXL',
    'VESPA VXL150':             'VESPA VXL',
    'VESPA 150VXL':             'VESPA VXL',
    'VESPA 150 VXL':            'VESPA VXL',
    'VESPITO 150':              'VESPITO',
    'VESPITO 150 ':             'VESPITO',
    'VESPIRO 150':              'VESPITO',
    'VESPITE':                  'VESPITO',
    'VESPIA 150':               'VESPA VXL',

    # ── Mercedes ACTROS ──────────────────────────────────────────────────────
    'MERCEDES E 53 AMG':        'E 53 AMG',

    # ── RANGE ROVER ──────────────────────────────────────────────────────────
    'RANGE ROVER VOG':          'RANGE ROVER VOGUE',
    'RANGE ROVER HSE':          'RANGE ROVER',

    # ── Cadillac ─────────────────────────────────────────────────────────────
    'CADILLAC ELDORADO':        'ELDORADO',
    'CADILLAC BROUGH':          'BROUGHAM',

    # ── PICKUP consolidation ──────────────────────────────────────────────────
    'DEKTE PICK UP TRUCK':      'D-MAX',    # Isuzu specific

    # ── FUSO (Mitsubishi) ─────────────────────────────────────────────────────
    'FUSO CANTER':              'CANTER',
    'FUSO FK260':               'FK260',
    'FUSO ROSA':                'ROSA',
    'FUSO BA':                  'CANTER',

    # ── misc fixes ───────────────────────────────────────────────────────────
    'SUPER 800 - 3':            'SUPER 800',
    'ACCORD CROSSTOUR':         'CROSSTOUR',
    'CAMAR':                    'CAMARO',
    'CRUIZ':                    'CRUZE',
    'MONJA':                    'MURANO',
    'LEYLAND':                  'PARTNER',
    'OUSTER':                   'OYSTER',
    'TASHOK LEYLAND FALCON':    'FALCON',
    'BAJAJ BALZAR':             'BALZAR',
    'BAJAJ C 150':              'C 150',
    'BAJAJ MINI 150':           'MINI 150',
    'BAJAJ PULSAR 150':         'PULSAR 150',
    'C 1025MY':                 'C 1025',
    'C 160MY':                  'C 160',
    'C 160M':                   'C 160',
    'C160MY':                   'C 160',
    'C1600':                    'C 1600',
    'ELGHT':                    'EIGHT',
    'FLHTCUI':                  'FLHTCU',
    'NCE UP':                   'NICE UP',
    'VELAIC':                   'VLEAICLE',  # garbled
    'VANDER HILL':              'VANDERHALL',
    'VANDERHALL CARM':          'VANDERHALL CARMEL',
    'VANDERHALL CARM':          'VANDERHALL CARMEL',
    'DEVILLE/COUPE':            'DEVILLE',
    'ELDORADO/COUPE':           'ELDORADO',
    'ELDORADO / COUPE':         'ELDORADO',
    'WRANGLER UNLIMITED':       'WRANGLER',
    'URBAN VAN':                'URVAN',
    'URVAN/COACH':              'URVAN',
    'URVAN/VAN':                'URVAN',
    '330-25':                   '',          # Abarth: not a valid model → junk
    'ES 3':                     'ES 3',      # Abarth ES3 concept — keep

    # Brand-only entries → junk
    'BAJAJA':                   '',
    'BAJAJ':                    '',
    'PHONDA':                   '',
    'THONDA':                   '',
    'BAJAL 15M.':               '',
    'HONDA':                    '',
    'BAJAJ-220CC':              'PULSAR 220',
    'BAJAJA':                   '',

    # ── SION IQ / SCION ───────────────────────────────────────────────────────
    'SCION IQ':                 'IQ',

    # ── SXL / SX normalization ────────────────────────────────────────────────
    'SXL 150':                  'SXL 150',
    'VESPA/150/MOTORCYCLE':     'VESPA 150',
    'VESPA VX':                 'VESPA VXL',

    # Ashok Leyland ── tighten
    'ASHOK LEYLAND FA':         'FALCON',
    'ASHOK LEYLAND':            '',

    # ── DFSK / FENGON normalization ───────────────────────────────────────────
    'FENGON GLORY 580':         'FENGON 580',
    'FENGON GLORY 600':         'FENGON 600',
    'FENGON GLORY IX7':         'FENGON IX7',
    'FENGON GLORY':             'FENGON',
    'FENGON C31':               'C31',
    'FENGON C32':               'C32',
    'PICK UP 32':               'C32',

    # ── CNI (Bajaj) ──────────────────────────────────────────────────────────
    'CNI 180':                  'C 180',
    'GRS 4531':                 '',          # not a real model

    # ── Honda specific ────────────────────────────────────────────────────────
    'FD 25 NT':                 'FORZA',
    'GPX DEMON150GR F1':        'GROM',      # GPX is a separate brand
    'CARROT ABYEI':             '',          # garbled
    'MAD':                      '',          # garbled
    'MRV':                      'CRV',       # typo
    'S-2000':                   'S2000',
    'S 2000':                   'S2000',
    'ZR V':                     'ZR-V',
    'HR (4WD)':                 'HR-V',
    'TAIGER':                   'TRIGGER',   # phonetic

    # ── VF series normalization (VinFast) ─────────────────────────────────────
    'SLINGSHOT SLINGS':         'SLINGSHOT',
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAKE-SPECIFIC CORRECTIONS
#
#  WHY: Some corrections are only valid for a particular make.
#  e.g. "ROCKY" means Daihatsu Rocky, not the same for every brand.
# ══════════════════════════════════════════════════════════════════════════════

MAKE_CORRECTIONS: dict[str, dict[str, str]] = {
    'DAIHATSU': {
        'DALTA': 'DELTA',
        'DAIHATSU DELTA': 'DELTA',
        'DAIHATSU FEROZA': 'FEROZA',
        'DAIHATSU TERIOS': 'TERIOS',
        'ROCKY': 'ROCKY',
    },
    'TOYOTA': {
        'FORTUNER': 'FORTUNER',
        'FORERUNNER': '4RUNNER',    # override global: Toyota → 4Runner
        'PRADO': 'LAND CRUISER PRADO',
        'TOKOMA': 'TACOMA',
        'TAKOMA': 'TACOMA',
        'TAKOMA PICK UP CARGO': 'TACOMA',
        'GRAND DELUX': 'LAND CRUISER',
        'GS 300': 'GS300',          # Note: GS300 is Lexus; Toyota lists it incorrectly
        'SY150T': '',               # motorcycle model, not Toyota
        'VESPA VXL': '',
        'VESPA VXL 150': '',
    },
    'NISSAN': {
        'CEDAR': 'CEDRIC',
        'CEDREC': 'CEDRIC',
        'CEDRICK': 'CEDRIC',
        'CENTRA': 'SENTRA',
        'NISSAN RICH': 'RICH',
        'TRANSPORTER T5': '',       # VW, not Nissan
        'SAXUS': '',                # not a real Nissan
    },
    'HONDA': {
        'ROCKY': '',                # not a Honda model
        'HAYABUSA': '',             # Suzuki, flag as wrong brand
        'YZF R1': '',
        'YZF R3': '',
        'YZF R6': '',
        'YZF R7': '',
        'MT-09': '',
        'YZFR7': '',
        'VESPA': '',
        'VESPA 150': '',
        'VESPA VXL': '',
        'VESPA VXL 150': '',
        'FLHCS': '',
        'FLHTK': '',
        'FLTRXSE': '',
        'ORN 160': 'UNICORN 160',
        'UNICORN QVN 1250': 'UNICORN',
        'NINED POE': 'NINJA',       # even here it's a wrong brand but clean it
        'PHONDA': '',
        'THONDA': '',
        'C B R 500': 'CBR 500',
        'C BR 500': 'CBR 500',
    },
    'BAJAJ': {
        'BAJAJ - 150': 'C 150',
        'BAJAJ 100CT': 'CT 100',
        'BAJAJ 150': 'C 150',
        'BAJAJ 150 CC': 'C 150',
        'BAJAJ 150BM': 'BM 150',
        'BAJAJ 150CC': 'C 150',
        'BAJAJ 180 CC': 'C 180',
        'BAJAJ 180CC': 'C 180',
        'BAJAJ 200CC': 'C 200',
        'BAJAJ 220 CC': 'C 220',
        'BAJAJ 220CC': 'C 220',
        'BAJAJ AVENGER': 'AVENGER',
        'BAJAJ DISCOVER': 'DISCOVER',
        'FLHTCUI': '',   # Harley under Bajaj
        'ACCORD': '',    # Honda under Bajaj
        'YZF R1': '',    # Yamaha under Bajaj
        'BMW': '',
    },
    'SUZUKI': {
        'NINJA 300': '',   # Kawasaki
        'HARLEY DAVIDSON': '',
        'JEEP': '',
    },
    'BMW': {
        'HARLEY DAVIDSON': '',
        'VESPA': '',
        'VESPA VXL': '',
        'VESPA VXL 150': '',
        'SCOMADI 2001': '',
        'SY150T': '',
        'SMART': '',
        'MINI COUNTRYMAN': 'MINI COUNTRYMAN',   # keep (BMW group)
    },
    'HYUNDAI': {
        'SLINGSHOT': '',
        'AVANTE': 'ELANTRA',
        'AVANTY': 'ELANTRA',
        'GENESIS COUPE': 'GENESIS COUPE',
        'GENES G90': 'GENESIS G90',
        'GRAND YUOR': 'GRANDEUR',
        'GRANDYUOR': 'GRANDEUR',
        'INDIAN SCOUT': '',         # Indian Motorcycle, not Hyundai
    },
    'KIA': {
        'YAMAHA MTN 890': '',
        'GAC - EMZOOM 1.5TG GS': '',
        'YONGO': 'BONGO',
        'MORNING': 'MORNING',
        'PEGAS': 'PICANTO',         # Kia Picanto sold as Pegas in some markets
    },
    'MITSUBISHI': {
        'CRAVAN MISTUBISHI': '',    # garbled brand
        'KAB 3': 'CANTER',
        'FUZO': 'FUSO',
        'ROZA': 'ROSA',
        'ROZA SW': 'ROSA',
        'ROZA/COACH': 'ROSA',
        'FOZO': 'FUSO',
    },
    'CHEVROLET': {
        'NINJA 300': '',
        'DAIMLER DOUBLE': '',       # garbled
        'DEFENDER': '',             # Land Rover
        'SMB': '',                  # garbled
        'SONEK SEDAN': '',          # garbled; possibly Sonic
    },
    'FORD': {
        'HURACAN': '',              # Lamborghini
        'CHALLENGER': 'BRONCO',     # Dodge/wrong → unclear, blank safer
        'LINCOLN': '',              # different brand
        'TRAVELLER': '',            # not a Ford model
        'TRETON': '',
        'VEGA': '',                 # Chevrolet Vega, not Ford
        'SKH F': '',
        'SKF': '',
    },
    'ISUZU': {
        'DEXER': 'D-MAX',
        'DEKTE PICK UP TRUCK': 'D-MAX',
    },
    'LAND ROVER': {
        'DEVENDER': 'DEFENDER',
        'DEVENDR': 'DEFENDER',
        'RANGE ROVER VOG': 'RANGE ROVER VOGUE',
        'RANGE': 'RANGE ROVER',
    },
    'HARLEY-DAVIDSON': {
        'DAVIDSON': '',
        'LF 150 2': '',             # not a Harley model
        'LF1502': '',
        'VULCAN (MOTORCYCLE)': '',  # Kawasaki
    },
    'PIAGGIO': {
        'HARLEY DAVIDSON': '',
        'CHIEFTAIN 1400': '',       # Indian Motorcycle
        'AYED250H7': '',
        'CHANGHUA CH150': '',
    },
    'MERCEDES BENZ': {
        'MUSTANG': '',              # Ford
        'CAMRY': '',                # Toyota
        'SISVO': 'S500',            # garbled
        'NIL': '',
        'NONE': '',
    },
    'LEXUS': {
        'HARLEY DAVIDSON FAT': '',
        'CH150T-34': '',
        'SCION IQ': 'IQ',
        'LUXGEN S5': '',            # Luxgen, not Lexus
        'SIXTIES 175': '',
        '200ZX (CAR)': '',          # Nissan 200ZX
    },
    'CHERY': {
        'NISSAN TIIDA': '',
        'ROCKY': '',
    },
    'GWM': {
        'HILUX': '',                # Toyota
        'LH2000 (MOTORCYCLE)': '',
        'MOTORCYCLE': '',
        'HAVAL H': 'H6',
    },
    'GEELY': {
        # Volvo (Geely parent) — we keep these
        'POLESTAR 2': 'POLESTAR 2',
        'POLESTAR 3': 'POLESTAR 3',
        'POLESTAR 4': 'POLESTAR 4',
    },
    'DFSK': {
        'FENGON GLORY 580': 'FENGON 580',
        'FENGON GLORY 600': 'FENGON 600',
        'FENGON GLORY IX7': 'FENGON IX7',
        'FENGON GLORY': 'FENGON',
    },
    'ZNA': {
        'YUMSUN': 'YUMSUN',         # ZNA Yumsun is a real model
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _translate_arabic(raw: str) -> tuple[str, bool]:
    """
    If raw contains Arabic/Farsi script, look it up in ARABIC_TRANSLATIONS.
    Returns (translated_string, was_translated).
    If not found in the table, returns (raw, False) — caller will mark ARABIC.
    """
    if not ARABIC_RE.search(raw):
        return raw, False
    stripped = raw.strip()
    translated = ARABIC_TRANSLATIONS.get(stripped)
    if translated is not None:
        return translated, True
    # Try case-insensitive key search
    for k, v in ARABIC_TRANSLATIONS.items():
        if k.strip() == stripped:
            return v, True
    return raw, False   # not found → caller handles


def classify(raw: str) -> str:
    """
    First-pass: determine if this value is usable at all.
    Returns a tag — only 'OK' values proceed to clean().
    """
    u = raw.strip().upper()
    if not u:                           return 'EMPTY'
    if DOLLAR_HASH_RE.match(u):         return 'DOLLAR_HASH'
    if PURE_DIGIT_RE.match(u):          return 'NUMBER_ONLY'
    if STANDALONE_YEAR_RE.match(u):     return 'YEAR_ONLY'
    if u in JUNK_SET:                   return 'JUNK'
    return 'OK'


def clean(raw: str, make: str = '') -> tuple[str, list[str]]:
    """
    Multi-step cleaner. Returns (canonical_model, [change_log]).
    Only called when classify() returns 'OK'.

    make: optional uppercase make name for make-specific corrections.
    """
    log: list[str] = []
    s = ' '.join(raw.strip().split())      # normalize whitespace
    if s != raw.strip():
        log.append('ws_normalized')

    upper = s.upper()

    # ── Step 1: direct correction-map lookup (make-specific first) ────────────
    if make:
        mc = MAKE_CORRECTIONS.get(make.upper(), {})
        if upper in mc:
            canon = mc[upper]
            if canon:
                log.append(f'make_corrected→{canon}')
                return canon, log
            else:
                return '', ['make_mapped_to_empty']

    if upper in CORRECTIONS:
        canon = CORRECTIONS[upper]
        if canon:
            log.append(f'corrected→{canon}')
            return canon, log
        else:
            return '', ['mapped_to_empty']

    # ── Step 2: strip known brand prefix ─────────────────────────────────────
    for brand in BRAND_PREFIXES:
        if upper.startswith(brand + ' ') or upper.startswith(brand + '-'):
            stripped = upper[len(brand):].lstrip('-').lstrip()
            if stripped and stripped not in BRAND_ONLY_ENTRIES:
                log.append(f'brand_stripped:{brand}')
                upper = stripped
                # Re-check corrections after brand strip
                if upper in CORRECTIONS:
                    canon = CORRECTIONS[upper]
                    if canon:
                        log.append(f'corrected_post_brand→{canon}')
                        return canon, log
            break

    # ── Step 3: strip body-type suffix after '/' or '(' ──────────────────────
    cleaned = BODY_SUFFIX_RE.sub('', upper).strip()
    if cleaned != upper:
        log.append('body_suffix_stripped')
        upper = cleaned

    # ── Step 4: strip trailing 4WD/2WD/AWD specifier ─────────────────────────
    cleaned = _4WD_RE.sub('', upper).strip()
    if cleaned != upper:
        log.append('drive_spec_stripped')
        upper = cleaned

    # ── Step 5: remove model-year tags ("26MY", "25Y") ───────────────────────
    cleaned = MY_TAG_RE.sub('', upper).strip()
    if cleaned != upper:
        log.append('MY_tag_removed')
        upper = cleaned

    # ── Step 6: remove embedded calendar years ────────────────────────────────
    cleaned = YEAR_RE.sub('', upper).strip()
    cleaned = ' '.join(cleaned.split())
    if cleaned != upper:
        log.append('calendar_year_removed')
        upper = cleaned

    # ── Step 7: remove CC/displacement specs ─────────────────────────────────
    cleaned = CC_RE.sub('', upper).strip()
    cleaned = ' '.join(cleaned.split())
    if cleaned != upper:
        log.append('cc_spec_removed')
        upper = cleaned

    # ── Step 8: second-pass correction check after all stripping ─────────────
    if make:
        mc = MAKE_CORRECTIONS.get(make.upper(), {})
        if upper in mc:
            canon = mc[upper]
            if canon:
                log.append(f'make_corrected_post_clean→{canon}')
                return canon, log
            else:
                return '', ['make_mapped_to_empty_post_clean']

    if upper in CORRECTIONS:
        canon = CORRECTIONS[upper]
        if canon:
            log.append(f'corrected_post_clean→{canon}')
            return canon, log
        else:
            return '', ['mapped_to_empty_post_clean']

    if not upper:
        return '', ['became_empty_after_cleaning']

    # Post-clean sanity: bare number after CC strip → junk
    if PURE_DIGIT_RE.match(upper):
        return '', ['became_number_after_cleaning']

    return upper, log


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def process_model(raw: str, make: str = '') -> dict:
    """
    Process a single raw model string.

    Parameters
    ----------
    raw  : the dirty model value
    make : the vehicle make (e.g. 'HONDA') — enables make-specific corrections
           and wrong-brand detection

    Returns a dict with keys:
        original    — unchanged input
        cleaned     — canonical model name (empty string if junk)
        status      — OK | CLEANED | JUNK | EMPTY | ARABIC | ARABIC_TRANSLATED |
                      NUMBER_ONLY | YEAR_ONLY | DOLLAR_HASH | WRONG_BRAND
        changes     — semicolon-separated log of transformations applied
    """
    raw = str(raw)
    make_upper = make.strip().upper() if make else ''

    # ── Step 0: Arabic transliteration ───────────────────────────────────────
    translated, was_translated = _translate_arabic(raw)
    if ARABIC_RE.search(raw):
        if was_translated:
            # The translation may itself be junk (e.g. "MOTORCYCLE")
            if translated == '' or classify(translated) != 'OK':
                return {
                    'original': raw,
                    'cleaned':  '',
                    'status':   'ARABIC_JUNK',
                    'changes':  f'arabic_translated:{translated or "empty"}',
                }
            # Continue with the translated string
            raw = translated
        else:
            return {
                'original': raw,
                'cleaned':  '',
                'status':   'ARABIC',
                'changes':  'arabic_untranslated',
            }

    tag = classify(raw)
    if tag != 'OK':
        return {'original': str(raw), 'cleaned': '', 'status': tag, 'changes': ''}

    # ── Step 0b: wrong-brand check ────────────────────────────────────────────
    upper_raw = raw.strip().upper()
    if make_upper:
        wrong = WRONG_BRAND_MODELS.get(make_upper, set())
        if upper_raw in wrong:
            return {
                'original': raw,
                'cleaned':  upper_raw,
                'status':   'WRONG_BRAND',
                'changes':  f'model_belongs_to_different_brand',
            }

    canon, log = clean(raw, make=make_upper)
    if was_translated:
        log = [f'arabic_translated:{translated}'] + log

    status = 'JUNK' if not canon else ('CLEANED' if log else 'OK')
    return {
        'original': raw,
        'cleaned':  canon,
        'status':   status,
        'changes':  '; '.join(log) if log else '',
    }


def process_dataframe(df, col: str, make_col: str = '') -> object:
    """
    Apply process_model() to every row in column `col`.
    Adds three new columns: model_clean, model_status, model_changes.
    Does NOT modify the original column.

    make_col: optional name of the 'make' column for make-aware cleaning.
    """
    import pandas as pd

    def _apply(row):
        make = str(row[make_col]) if make_col and make_col in row else ''
        return process_model(str(row[col]), make=make)

    results = df.apply(_apply, axis=1)
    df = df.copy()
    df['model_clean']   = [r['cleaned'] for r in results]
    df['model_status']  = [r['status']  for r in results]
    df['model_changes'] = [r['changes'] for r in results]
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Clean vehicle model column for VIN detection datasets (v2).'
    )
    parser.add_argument('--input',   required=True,  help='Input CSV path')
    parser.add_argument('--col',     default='Model', help='Model column name (default: Model)')
    parser.add_argument('--make',    default='',      help='Make column name for make-aware cleaning (optional)')
    parser.add_argument('--output',  required=True,  help='Output cleaned CSV path')
    parser.add_argument('--mapping', default='model_mapping.csv',
                        help='Output unique original→cleaned mapping CSV (default: model_mapping.csv)')
    parser.add_argument('--no-header', action='store_true',
                        help='Input CSV has no header row')
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit('pandas is required: pip install pandas')

    print(f'Reading  : {args.input}')
    header = None if args.no_header else 'infer'
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, header=header)

    if args.col not in df.columns:
        print(f'ERROR: column "{args.col}" not found.')
        print(f'Available columns: {list(df.columns)}')
        sys.exit(1)

    if args.make and args.make not in df.columns:
        print(f'WARNING: make column "{args.make}" not found. Running without make context.')
        args.make = ''

    print(f'Rows     : {len(df):,}')
    make_info = f' (make-aware via column "{args.make}")' if args.make else ''
    print(f'Cleaning column "{args.col}"{make_info} ...\n')
    df_out = process_dataframe(df, args.col, make_col=args.make)

    # ── summary ───────────────────────────────────────────────────────────────
    counts = df_out['model_status'].value_counts()
    total  = len(df_out)
    print('─' * 48)
    print(f'{"STATUS":<25} {"COUNT":>7}  {"% of total":>10}')
    print('─' * 48)
    for status, n in counts.items():
        print(f'{status:<25} {n:>7,}  {n/total*100:>9.1f}%')
    print('─' * 48)
    junk_statuses = {'JUNK','EMPTY','ARABIC','ARABIC_JUNK','NUMBER_ONLY',
                     'YEAR_ONLY','DOLLAR_HASH','WRONG_BRAND'}
    junk_count = sum(counts.get(t, 0) for t in junk_statuses)
    print(f'{"Total junk/discarded":<25} {junk_count:>7,}  {junk_count/total*100:>9.1f}%')
    print(f'{"Usable rows":<25} {total-junk_count:>7,}  {(total-junk_count)/total*100:>9.1f}%')
    print()

    unique_before = df_out[args.col].nunique()
    usable_mask   = ~df_out['model_status'].isin(junk_statuses)
    unique_after  = df_out.loc[usable_mask, 'model_clean'].nunique()
    print(f'Unique model values  BEFORE: {unique_before:,}')
    print(f'Unique model values  AFTER : {unique_after:,}  (usable rows only)')
    print()

    df_out.to_csv(args.output, index=False)
    print(f'Cleaned dataset  → {args.output}')

    key_cols = [args.col, 'model_clean', 'model_status', 'model_changes']
    if args.make:
        key_cols = [args.make] + key_cols
    mapping = (
        df_out[key_cols]
        .drop_duplicates(subset=[args.col])
        .sort_values(args.col)
    )
    mapping.to_csv(args.mapping, index=False)
    print(f'Model mapping    → {args.mapping}')
    print()
    print('TIP: Review model_mapping.csv — check WRONG_BRAND rows and')
    print('     ARABIC_UNTRANSLATED rows, and add any remaining corrections')
    print('     to MAKE_CORRECTIONS or CORRECTIONS as appropriate.')


if __name__ == '__main__':
    main()
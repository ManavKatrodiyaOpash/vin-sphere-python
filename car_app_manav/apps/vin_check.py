# =====================================================
# VIN CHECK DIGIT VALIDATOR
# =====================================================

VIN_VALUES = {
    'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,
    'J':1,'K':2,'L':3,'M':4,'N':5,       'P':7,'R':9,
           'S':2,'T':3,'U':4,'V':5,'W':6,'X':7,'Y':8,'Z':9,
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
}

VIN_WEIGHTS = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

YEAR_MAP = {
    "V":1997, "W":1998, "X":1999, "Y":2000, "1":2001, "2":2002, "3":2003, "4":2004,
    "5":2005, "6":2006, "7":2007, "8":2008, "9":2009, "A":2010, "B":2011, "C":2012,
    "D":2013, "E":2014, "F":2015, "G":2016, "H":2017, "J":2018, "K":2019, "L":2020,
    "M":2021, "N":2022, "P":2023, "R":2024, "S":2025, "T":2026
}

def validate_check_digit(vin):
    """Returns True if check digit (pos9) is valid."""
    try:
        total = sum(VIN_VALUES.get(c, 0) * VIN_WEIGHTS[i] for i, c in enumerate(vin))
        remainder = total % 11
        expected = 'X' if remainder == 10 else str(remainder)
        return vin[8] == expected, expected
    except Exception:
        return False, "?"

def validate_vin_chars(vin):
    """VIN cannot contain I, O, Q."""
    bad = [c for c in vin if c in ('I', 'O', 'Q')]
    return len(bad) == 0, bad
# Complete VIN Decoder System

## Overview

A **VIN (Vehicle Identification Number)** is a globally standardized **17-character alphanumeric identifier** assigned to every motor vehicle.

VINs are regulated by:

* ISO 3779
* ISO 3780
* NHTSA (United States)
* SAE Standards
* Regional automotive authorities

A VIN provides detailed information about a vehicle including:

* Manufacturer
* Country of origin
* Vehicle type
* Model
* Engine
* Body style
* Transmission
* Safety systems
* Model year
* Manufacturing plant
* Production sequence number

---

# VIN Structure

Example VIN:

```text
1HGCM82633A123456
```

VIN Layout:

```text
1HG CM826 3 3 A 123456
│   │     │ │ │   │
│   │     │ │ │   └── Serial Number
│   │     │ │ └────── Manufacturing Plant
│   │     │ └──────── Model Year
│   │     └────────── Check Digit
│   └──────────────── Vehicle Descriptor Section (VDS)
└──────────────────── World Manufacturer Identifier (WMI)
```

---

# VIN Sections

| Section | Positions | Name                          |
| ------- | --------- | ----------------------------- |
| WMI     | 1-3       | World Manufacturer Identifier |
| VDS     | 4-9       | Vehicle Descriptor Section    |
| VIS     | 10-17     | Vehicle Identifier Section    |

---

# Section 1: WMI (World Manufacturer Identifier)

## Position 1 — Country / Region

Determines manufacturing region.

| Code | Country          |
| ---- | ---------------- |
| 1    | USA              |
| 4    | USA              |
| 5    | USA              |
| 2    | Canada           |
| 3    | Mexico           |
| J    | Japan            |
| K    | South Korea      |
| W    | Germany          |
| Z    | Italy            |
| S    | United Kingdom   |
| Y    | Sweden / Finland |
| L    | China            |
| M    | India            |
| V    | France / Spain   |
| T    | Switzerland      |
| R    | Taiwan           |
| 9    | Brazil           |

Example:

```text
J = Japan
W = Germany
M = India
```

---

## Position 2 — Manufacturer

| Code | Manufacturer   |
| ---- | -------------- |
| H    | Honda          |
| T    | Toyota         |
| N    | Nissan         |
| B    | BMW            |
| A    | Audi           |
| M    | Mercedes-Benz  |
| F    | Ford           |
| G    | General Motors |
| C    | Chrysler       |
| V    | Volkswagen     |
| P    | Porsche        |
| L    | Lexus          |

---

## Position 3 — Vehicle Division / Type

Manufacturer-specific.

Examples:

| Code | Meaning            |
| ---- | ------------------ |
| A    | Passenger Vehicle  |
| B    | Bus                |
| C    | Commercial Vehicle |
| T    | Truck              |
| U    | SUV                |
| V    | Van                |

Example:

```text
WAU

W = Germany
A = Audi
U = Passenger Vehicle
```

---

# Section 2: VDS (Vehicle Descriptor Section)

Positions 4-9

Contains:

* Model
* Engine
* Trim
* Body style
* Safety systems
* Transmission
* Drive type

---

## Position 4 — Body Type

| Code | Body Style  |
| ---- | ----------- |
| S    | Sedan       |
| H    | Hatchback   |
| C    | Coupe       |
| U    | SUV         |
| P    | Pickup      |
| W    | Wagon       |
| V    | Van         |
| K    | Convertible |

---

## Position 5 — Engine Type

Examples:

| Code | Engine         |
| ---- | -------------- |
| A    | 1.5L Petrol    |
| B    | 2.0L Petrol    |
| C    | 3.0L Petrol    |
| D    | Diesel         |
| H    | Hybrid         |
| E    | Electric       |
| T    | Turbocharged   |
| P    | Plug-In Hybrid |

**Note:** Engine decoding is OEM-specific.

---

## Position 6 — Safety Systems

| Code | Description         |
| ---- | ------------------- |
| A    | Driver Airbag       |
| B    | Dual Airbags        |
| C    | ABS                 |
| D    | Side Airbags        |
| E    | Curtain Airbags     |
| F    | Full Safety Package |

---

## Position 7 — Transmission / Drive Type

| Code | Transmission   |
| ---- | -------------- |
| A    | Manual         |
| B    | Automatic      |
| C    | CVT            |
| D    | AWD            |
| E    | 4WD            |
| F    | Dual Clutch    |
| G    | Electric Drive |

---

## Position 8 — Trim / Series

| Code | Trim        |
| ---- | ----------- |
| B    | Base        |
| S    | Sport       |
| X    | Executive   |
| L    | Luxury      |
| F    | Full Option |
| P    | Premium     |

---

## Position 9 — Check Digit

Used for VIN validation.

### Transliteration Table

```text
A = 1
B = 2
C = 3
D = 4
E = 5
F = 6
G = 7
H = 8
J = 1
K = 2
L = 3
M = 4
N = 5
P = 7
R = 9
S = 2
T = 3
U = 4
V = 5
W = 6
X = 7
Y = 8
Z = 9
```

---

### Position Weights

```text
8 7 6 5 4 3 2 10 0 9 8 7 6 5 4 3 2
```

---

### Formula

```text
Check Digit =
(sum(character_value × weight)) mod 11
```

Result:

```text
0-9 = Numeric value
10 = X
```

---

# Section 3: VIS (Vehicle Identifier Section)

Positions 10-17

Contains:

* Model Year
* Manufacturing Plant
* Serial Number

---

## Position 10 — Model Year

### Year Code Mapping

| Code | Year |
| ---- | ---- |
| A    | 1980 |
| B    | 1981 |
| C    | 1982 |
| D    | 1983 |
| E    | 1984 |
| F    | 1985 |
| G    | 1986 |
| H    | 1987 |
| J    | 1988 |
| K    | 1989 |
| L    | 1990 |
| M    | 1991 |
| N    | 1992 |
| P    | 1993 |
| R    | 1994 |
| S    | 1995 |
| T    | 1996 |
| V    | 1997 |
| W    | 1998 |
| X    | 1999 |
| Y    | 2000 |
| 1    | 2001 |
| 2    | 2002 |
| 3    | 2003 |
| 4    | 2004 |
| 5    | 2005 |
| 6    | 2006 |
| 7    | 2007 |
| 8    | 2008 |
| 9    | 2009 |

Cycle repeats every 30 years.

Examples:

```text
A = 1980, 2010, 2040
B = 1981, 2011, 2041
```

Context is required for accurate year determination.

---

## Position 11 — Manufacturing Plant

OEM-specific.

Examples:

| Code | Plant          |
| ---- | -------------- |
| A    | Alabama        |
| B    | Berlin         |
| J    | Japan Plant    |
| M    | Munich         |
| T    | Texas          |
| D    | Dubai Assembly |

---

## Positions 12-17 — Serial Number

Unique production sequence.

Example:

```text
123456
```

Uses:

* Warranty tracking
* Recall management
* Insurance verification
* Vehicle registration
* Production sequencing

---

# Example VIN Decoding

VIN:

```text
WBA8E9G50GNU12345
```

---

## Decoding Result

| Position | Value  | Meaning        |
| -------- | ------ | -------------- |
| 1        | W      | Germany        |
| 2        | B      | BMW            |
| 3        | A      | Passenger Car  |
| 4        | 8      | BMW 3 Series   |
| 5        | E      | Sedan          |
| 6        | 9      | Safety Package |
| 7        | G      | Automatic      |
| 8        | 5      | Luxury Trim    |
| 9        | 0      | Check Digit    |
| 10       | G      | 2016           |
| 11       | N      | Munich Plant   |
| 12-17    | U12345 | Serial Number  |

---

# VIN Decoder API Design

## Request

```json
{
  "vin": "WBA8E9G50GNU12345"
}
```

---

## Response

```json
{
  "vin": "WBA8E9G50GNU12345",
  "valid": true,
  "country": "Germany",
  "manufacturer": "BMW",
  "vehicle_type": "Passenger Car",
  "model": "3 Series",
  "body_type": "Sedan",
  "engine": "2.0L Turbo",
  "transmission": "Automatic",
  "trim": "Luxury",
  "year": 2016,
  "plant": "Munich",
  "serial_number": "U12345"
}
```

---

# VIN Decoder Database Schema

## Manufacturers

```json
{
  "wmi": "WBA",
  "manufacturer": "BMW",
  "country": "Germany"
}
```

---

## Models

```json
{
  "manufacturer": "BMW",
  "code": "8",
  "model": "3 Series"
}
```

---

## Engines

```json
{
  "manufacturer": "BMW",
  "code": "E",
  "engine": "2.0L Turbo"
}
```

---

# Production Architecture

```text
User
  ↓
API Gateway
  ↓
VIN Validation Service
  ↓
WMI Decoder
  ↓
VDS Decoder
  ↓
VIS Decoder
  ↓
Manufacturer Rules Engine
  ↓
Response Builder
  ↓
Client
```

---

# Advanced VIN Decoder Features

## OEM Rule Engine

Manufacturer-specific decoding:

* Toyota
* Honda
* BMW
* Audi
* Mercedes-Benz
* Nissan
* Hyundai
* Ford
* Chevrolet
* Tesla

Each manufacturer has unique mappings.

---

## VIN Validation

Checks:

* Length = 17
* Invalid characters excluded

```text
I
O
Q
```

* Check digit verification
* WMI verification

---

## Vehicle Intelligence Features

Generate:

* Market value estimation
* Recall information
* Insurance classification
* Fuel economy
* Safety ratings
* Vehicle history integration
* Parts compatibility
* Maintenance recommendations

---

# Enterprise-Scale VIN Decoder

### Supported Data Sources

* NHTSA VIN Database
* WMI Registry
* Manufacturer Catalogs
* Vehicle Production Databases
* Insurance Vehicle Registries

### Scale Targets

* 100M+ VIN records
* Sub-100ms decode time
* Multi-region deployment
* OEM-specific rule processing
* Real-time validation

---

# Final Output Example

```json
{
  "vin": "WBA8E9G50GNU12345",
  "valid": true,
  "country": "Germany",
  "manufacturer": "BMW",
  "model": "3 Series",
  "year": 2016,
  "engine": "2.0L Turbo",
  "body_type": "Sedan",
  "transmission": "Automatic",
  "trim": "Luxury",
  "plant": "Munich",
  "serial_number": "U12345",
  "vehicle_type": "Passenger Car",
  "safety_package": "Advanced Safety Package"
}
```

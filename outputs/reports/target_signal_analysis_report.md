# Automatic Target Signal Analysis Report

Analyzed a representative sample of **100,000** valid records.

## Positional Signal Strength Table

| Target Column | Type | Cardinality | Missing % | Max Cramer's V | Best Position | Signal Status |
| --- | --- | --- | --- | --- | --- | --- |
| `make` | Classification | 217 | 0.00% | 0.620 | Position 2 | **STRONG** |
| `model` | Classification | 3,812 | 0.60% | 0.818 | Position 3 | **STRONG** |
| `trim` | Classification | 2,624 | 2.15% | 0.569 | Position 3 | **STRONG** |
| `bodyType` | Classification | 31 | 0.00% | 0.246 | Position 3 | **STRONG** |
| `year` | Classification | 59 | 0.00% | 0.922 | Position 10 | **STRONG** |
| `cylinders` | Classification | 9 | 1.75% | 0.338 | Position 3 | **STRONG** |
| `origin` | Classification | 124 | 0.00% | 0.716 | Position 1 | **STRONG** |
| `noOfPassengers` | Classification | 66 | 0.00% | 0.178 | Position 3 | **STRONG** |
| `weightInKg` | Classification | 2,602 | 0.18% | 0.311 | Position 2 | **STRONG** |
| `regionalSpec` | Classification | 2 | 0.02% | 0.127 | Position 3 | **STRONG** |
| `color` | Classification | 173 | 6.46% | 0.085 | Position 3 | **WEAK** |

## Signal Warnings
The following targets show **weak structural signals** in the VIN. Predictions for them may be less reliable:
- :warning: Target 'color' has weak position association (Max Cramer's V = 0.085 at Position 3).
import logging
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def hamming_distance(s1: str, s2: str) -> int:
    min_len = min(len(s1), len(s2))
    dist = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
    dist += abs(len(s1) - len(s2))
    return dist

def prefix_similarity(s1: str, s2: str) -> int:
    common_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            common_len += 1
        else:
            break
    return common_len

class PrefixSimilarityEngine:
    def __init__(self):
        self.train_chassis = []
        self.train_df = None
        self.prefix_stats = {}
        self.targets = ["make", "model", "trim", "body_type", "origin", "regional_specs"]
        self.csv_cols = {
            "make": "make",
            "model": "model",
            "trim": "trim",
            "body_type": "bodyType",
            "origin": "origin",
            "regional_specs": "regionalSpec"
        }

    def fit(self, df: pd.DataFrame):
        """
        Fits the engine on the training DataFrame:
        1. Saves training chassis numbers and their target attributes.
        2. Calculates Prefix Intelligence distributions for prefix_2 to prefix_7.
        """
        self.train_df = df.copy()
        
        # Clean chassisNumber and targets
        self.train_df['chassisNumber'] = self.train_df['chassisNumber'].astype(str).str.upper().str.strip()
        for target, col in self.csv_cols.items():
            if col in self.train_df.columns:
                self.train_df[col] = self.train_df[col].astype(str).str.upper().str.strip()
                
        self.train_chassis = self.train_df['chassisNumber'].tolist()
        
        # Build self.train_records list of dicts for fast attribute lookup
        self.train_records = []
        self.prefix_records = {}
        for _, row in self.train_df.iterrows():
            rec = {
                'chassisNumber': row['chassisNumber']
            }
            for target, col in self.csv_cols.items():
                rec[target] = row[col] if (col in row and pd.notna(row[col])) else "UNKNOWN"
            try:
                rec['year'] = int(float(row['year'])) if ('year' in row and pd.notna(row['year']) and str(row['year']).lower() != 'nan') else 0
            except ValueError:
                rec['year'] = 0
            try:
                rec['weight'] = float(row['weightInKg']) if ('weightInKg' in row and pd.notna(row['weightInKg']) and str(row['weightInKg']).lower() != 'nan') else 0.0
            except ValueError:
                rec['weight'] = 0.0
            self.train_records.append(rec)
            
            # Index by prefix lengths 3 and 2 for fast similarity pre-filtering
            chassis_num = rec['chassisNumber']
            for length in [3, 2]:
                pref = chassis_num[:length]
                if length not in self.prefix_records:
                    self.prefix_records[length] = {}
                if pref not in self.prefix_records[length]:
                    self.prefix_records[length][pref] = []
                self.prefix_records[length][pref].append(rec)
            
        # Calculate Prefix Intelligence
        self.prefix_stats = {}
        for length in range(2, 8):
            self.prefix_stats[length] = {}
            prefix_col = f"prefix_{length}"
            temp_df = self.train_df.copy()
            temp_df[prefix_col] = temp_df['chassisNumber'].str[:length]
            
            # Find the mode (most common value) for each target
            for prefix_val, group in temp_df.groupby(prefix_col):
                prefix_entry = {}
                for target, col in self.csv_cols.items():
                    if col in group.columns:
                        valid_group = group[group[col].notna() & (~group[col].isin(["NAN", "NONE", "", "UNKNOWN"]))]
                        if len(valid_group) > 0:
                            prefix_entry[target] = valid_group[col].mode().iloc[0]
                        else:
                            prefix_entry[target] = "UNKNOWN"
                    else:
                        prefix_entry[target] = "UNKNOWN"
                self.prefix_stats[length][prefix_val] = prefix_entry

        return self

    def find_nearest_neighbors(self, chassis: str, top_n: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Finds the nearest neighbors for a given chassis number using prefix pre-filtering.
        """
        chassis = chassis.upper().strip()
        
        # 1. Candidate pre-filtering based on prefix_3
        candidates = []
        pref_3 = chassis[:3]
        if 3 in self.prefix_records and pref_3 in self.prefix_records[3]:
            candidates = self.prefix_records[3][pref_3]
            
        # 2. If empty, fall back to prefix_2
        if len(candidates) == 0:
            pref_2 = chassis[:2]
            if 2 in self.prefix_records and pref_2 in self.prefix_records[2]:
                candidates = self.prefix_records[2][pref_2]
                
        # 3. Fallback to entire training set only if still completely empty
        if len(candidates) == 0:
            candidates = self.train_records
            
        results = []
        for rec in candidates:
            other_chassis = rec['chassisNumber']
            
            # Compute distances
            lev = levenshtein_distance(chassis, other_chassis)
            ham = hamming_distance(chassis, other_chassis)
            pref = prefix_similarity(chassis, other_chassis)
            
            # Composite distance
            distance = lev + 0.5 * ham - 0.2 * pref
            
            attributes = {
                'year': rec['year'],
                'weight': rec['weight']
            }
            for target in self.targets:
                attributes[target] = rec[target]
                
            results.append((other_chassis, distance, attributes))
            
        # Sort by distance ascending
        results = sorted(results, key=lambda x: x[1])
        return results[:top_n]

    def get_prefix_intel(self, chassis: str) -> Dict[str, Any]:
        """
        Looks up prefix intelligence for the longest matching prefix (length 7 down to 2).
        """
        chassis = chassis.upper().strip()
        for length in range(7, 1, -1):
            pref_val = chassis[:length]
            if length in self.prefix_stats and pref_val in self.prefix_stats[length]:
                return self.prefix_stats[length][pref_val]
        
        # Default fallback
        return {target: "UNKNOWN" for target in self.targets}

    def transform(self, chassis_series: pd.Series) -> pd.DataFrame:
        """
        Transforms a Series of chassis numbers into similarity search and prefix intelligence features.
        Features created:
        - sim_make_1, sim_model_1, sim_trim_1, sim_year_1, sim_score_1 (up to top 3 neighbors)
        - pref_intel_make, pref_intel_model, pref_intel_trim, etc.
        """
        logger.info("Generating similarity search and prefix intelligence features...")
        output_features = []
        
        for idx, chassis in enumerate(chassis_series):
            chassis_clean = str(chassis).upper().strip()
            row_feat = {}
            
            # 1. Prefix Intelligence features
            intel = self.get_prefix_intel(chassis_clean)
            for target in self.targets:
                row_feat[f"pref_intel_{target}"] = intel[target]
                
            # 2. Similarity search features (Top 3 neighbors)
            neighbors = self.find_nearest_neighbors(chassis_clean, top_n=3)
            for n_idx, (nb_chassis, dist, nb_attrs) in enumerate(neighbors):
                row_feat[f"sim_score_{n_idx+1}"] = float(dist)
                row_feat[f"sim_chassis_{n_idx+1}"] = nb_chassis
                for target in self.targets:
                    row_feat[f"sim_{target}_{n_idx+1}"] = nb_attrs[target]
                row_feat[f"sim_year_{n_idx+1}"] = str(nb_attrs['year'])
                row_feat[f"sim_weight_{n_idx+1}"] = float(nb_attrs['weight'])
                
            output_features.append(row_feat)
            
        return pd.DataFrame(output_features, index=chassis_series.index)

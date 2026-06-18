import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root and local folder to sys.path
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.append(str(_parent))

def run_analysis(data_path: str, output_dir: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Normalize chassis numbers and categorical columns
    df['chassisNumber'] = df['chassisNumber'].astype(str).str.upper().str.strip()
    
    categorical_targets = {
        'make': 'make',
        'model': 'model',
        'trim': 'trim',
        'body_type': 'bodyType',
        'origin': 'origin',
        'regional_spec': 'regionalSpec'
    }
    
    for key, col in categorical_targets.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip().fillna("UNKNOWN")
            
    # Uniqueness Statistics
    total_records = len(df)
    unique_vins = df['chassisNumber'].nunique()
    print(f"Total records: {total_records}")
    print(f"Unique chassis numbers: {unique_vins}")
    
    # Analyze prefix purity for prefix_3, prefix_4, prefix_5, prefix_6
    purity_results = {}
    prefix_stats = {}
    
    for length in [3, 4, 5, 6]:
        df[f'prefix_{length}'] = df['chassisNumber'].apply(lambda x: x[:length])
        purity_results[length] = {}
        
        for key, col in categorical_targets.items():
            puries = []
            grouped = df.groupby(f'prefix_{length}')[col]
            for pref, group in grouped:
                if len(group) == 0:
                    continue
                counts = group.value_counts()
                dominant_val = counts.index[0]
                dominant_count = counts.iloc[0]
                purity = dominant_count / len(group)
                puries.append(purity)
            
            purity_results[length][key] = np.mean(puries) if puries else 0.0

    # Generate report showing prefix statistics and build prefix intelligence engine dict
    for length in [3, 4, 5, 6]:
        prefix_stats[length] = {}
        grouped = df.groupby(f'prefix_{length}')
        for pref, group in grouped:
            stats = {}
            for key, col in categorical_targets.items():
                counts = group[col].value_counts()
                if not counts.empty:
                    stats[f'most_common_{key}'] = counts.index[0]
                    stats[f'{key}_confidence'] = float(counts.iloc[0] / len(group))
                else:
                    stats[f'most_common_{key}'] = "UNKNOWN"
                    stats[f'{key}_confidence'] = 0.0
            
            stats['confidence_score'] = (stats['make_confidence'] + stats['model_confidence']) / 2.0
            prefix_stats[length][pref] = stats

    # Save prefix_statistics.pkl
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, 'prefix_statistics.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(prefix_stats, f)
    print(f"Saved prefix statistics to {pkl_path}")
    
    # Generate the Markdown Report
    report_content = []
    report_content.append("# 12-Character VIN Dataset Pattern & Prefix Purity Report\n")
    report_content.append(f"**Total Records:** {total_records}  ")
    report_content.append(f"**Unique VINs:** {unique_vins}  \n")
    
    report_content.append("## Average Prefix Purity by Length and Target Attribute\n")
    report_content.append("| Prefix Length | Make Purity | Model Purity | Trim Purity | Origin Purity | Regional Spec Purity |")
    report_content.append("|---|---|---|---|---|---|")
    for length in [3, 4, 5, 6]:
        r = purity_results[length]
        report_content.append(f"| prefix_{length} | {r['make']:.2%} | {r['model']:.2%} | {r['trim']:.2%} | {r['origin']:.2%} | {r['regional_spec']:.2%} |")
    
    report_content.append("\n## Most Common Attributes by Prefix (Top Prefixes by Count)\n")
    for length in [3, 4, 5, 6]:
        report_content.append(f"### Prefix Length {length} (Top 10 Prefixes)")
        report_content.append("| Prefix | Count | Most Common Make (Conf) | Most Common Model (Conf) | Most Common Trim (Conf) | Most Common Origin (Conf) |")
        report_content.append("|---|---|---|---|---|---|")
        
        pref_counts = df[f'prefix_{length}'].value_counts().head(10)
        for pref, count in pref_counts.items():
            stats = prefix_stats[length][pref]
            make_str = f"{stats['most_common_make']} ({stats['make_confidence']:.0%})"
            model_str = f"{stats['most_common_model']} ({stats['model_confidence']:.0%})"
            trim_str = f"{stats['most_common_trim']} ({stats['trim_confidence']:.0%})"
            origin_str = f"{stats['most_common_origin']} ({stats['origin_confidence']:.0%})"
            report_content.append(f"| {pref} | {count} | {make_str} | {model_str} | {trim_str} | {origin_str} |")
        report_content.append("")
        
    report_path = os.path.join(output_dir, 'dataset_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    print(f"Generated report at {report_path}")

if __name__ == "__main__":
    data_path = os.path.join(_parent, 'Data', 'final_clean_12.csv')
    output_dir = os.path.join(_parent, 'chat_cat_short_vin_12', 'models')
    run_analysis(data_path, output_dir)

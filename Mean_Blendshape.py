import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"D:\Intern work\Data for research\data_rearranged"
OUTLIER_LOG_FILE = "outliers_log.csv"

GROUP_MAPPING = {
    'Group_A': ['pr_8', 'pr_17', 'pr_15', 'pr_4', 'pr_1', 'pr_22'],
    'Group_B': ['pr_9', 'pr_23', 'pr_3', 'pr_10', 'pr_21', 'pr_11']
}

EXPRESSIONS = [
    'Attentional_Engagement',
    'Aversion',
    'Concentration',
    'Dejection',
    'Positive_Social_Expression',
    'Skepticism',
    'Startle_Response',
    'Tension_Stress'
]

def run_numerical_aggregation(target_group):
    print(f"\n⚡ Starting Matrix Aggregation Pipeline for {target_group}...")
    
    # Load safe isolation keys: (Participant, Event_ID) tuple pairings
    outliers = set()
    if os.path.exists(OUTLIER_LOG_FILE):
        try:
            df_out = pd.read_csv(OUTLIER_LOG_FILE)
            df_active_outliers = df_out[df_out['Status'] == 'Outlier']
            outliers = set(zip(df_active_outliers['Participant'].astype(str), df_active_outliers['Event_ID'].astype(str)))
            print(f" Loaded {len(outliers)} total global active outlier rules.")
        except Exception as e:
            print(f"⚠️ Note loading outlier tracking file: {e}")
            
    participants = sorted(GROUP_MAPPING[target_group], key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
    all_individual_data = {}
    
    # DYNAMIC OUTPUT PATH SEPARATION
    output_dir = os.path.join(BASE_DIR, "Groups", target_group)
    os.makedirs(output_dir, exist_ok=True) # Ensures the folder exists before writing
    output_excel_path = os.path.join(output_dir, f"{target_group}_Hierarchical_Blendshape_Averages.xlsx")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for p_id in participants:
            csv_path = os.path.join(BASE_DIR, "Groups", target_group, "data", f"{p_id}.csv")
            evidence_path = os.path.join(BASE_DIR, "Groups", target_group, "data", f"{p_id}_event_evidence.csv")
            
            if not os.path.exists(csv_path) or not os.path.exists(evidence_path):
                print(f"⚠️ Skipping {p_id}: Missing raw maps or target evidence logs.")
                continue
                
            df_shapes = pd.read_csv(csv_path)
            df_ev = pd.read_csv(evidence_path)
            
            p_rows = []
            for expr in EXPRESSIONS:
                expr_csv_name = expr.replace('_', ' ')
                if expr == 'Tension_Stress':
                    expr_csv_name = 'Tension/Stress'
                    
                df_ev_filtered = df_ev[df_ev['Expression'] == expr_csv_name].head(100)
                
                valid_frame_indices = []
                for _, row_ev in df_ev_filtered.iterrows():
                    ev_id = row_ev['Event ID']
                    
                    # SAFE COMPOSITE VERIFICATION LAYER
                    if (str(p_id), str(ev_id)) not in outliers:
                        valid_frame_indices.append(int(row_ev['Frame Index']))
                        
                blendshape_cols = df_shapes.columns[3:]  
                
                if valid_frame_indices:
                    valid_frame_indices = [f for f in valid_frame_indices if f < len(df_shapes)]
                    df_valid_instances = df_shapes.iloc[valid_frame_indices]
                    
                    mean_vector = df_valid_instances[blendshape_cols].mean()
                    mean_vector.name = expr
                    p_rows.append(mean_vector)
                else:
                    mean_vector = pd.Series(0.0, index=blendshape_cols, name=expr)
                    p_rows.append(mean_vector)
                    
            df_p_summary = pd.DataFrame(p_rows)
            df_p_summary.index.name = "Expression"
            
            all_individual_data[p_id] = df_p_summary
            df_p_summary.to_excel(writer, sheet_name=p_id)
            print(f"    Compiled {p_id} Sheet (8 rows of expressions × {len(blendshape_cols)} blendshapes)")
            
        if all_individual_data:
            stacked_df = pd.concat(all_individual_data.values(), keys=all_individual_data.keys())
            group_avg_df = stacked_df.groupby(level=1).mean().reindex(EXPRESSIONS)
            group_avg_df.index.name = "Expression"
            
            group_avg_df.to_excel(writer, sheet_name="Group_Average")
            print(f"Unified Cohort Average complete! Sheet 'Group_Average' generated.")
            print(f"Workbook completely rendered at: {output_excel_path}")
        else:
            print("❌ Failure Matrix: No data could be processed.")

if __name__ == "__main__":
    print("      Average Blendshape Aggregation Pipeline started...      ")
    
    # Process both groups sequentially with a single click
    run_numerical_aggregation('Group_A')
    run_numerical_aggregation('Group_B')
    
    print("\n All operations completed successfully.")
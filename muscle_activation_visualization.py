import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"D:\Intern work\Data for research\data_rearranged"
GROUPS = ['Group_A', 'Group_B']

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

def batch_generate_equalizer_plots():
    print("🚀 Starting Equalizer Musculature Spectrum Pipeline...\n")
    
    for group in GROUPS:
        group_dir = os.path.join(BASE_DIR, "Groups", group)
        file_name = f"{group}_Hierarchical_Blendshape_Averages.xlsx"
        workbook_path = os.path.join(group_dir, file_name)
        
        if not os.path.exists(workbook_path):
            print(f"⚠️ Skipping {group}: Target workbook not found.")
            continue
            
        print(f"📦 Processing Cohort: {file_name}")
        
        output_plot_dir = os.path.join(group_dir, "Muscle Activation Plots")
        os.makedirs(output_plot_dir, exist_ok=True)
        
        try:
            df = pd.read_excel(workbook_path, sheet_name="Group_Average")
        except Exception as e:
            print(f"❌ Error loading sheet 'Group_Average' from {file_name}: {e}")
            continue
            
        if 'Expression' not in df.columns:
            print(f"❌ Structural Error: 'Expression' column missing.")
            continue
            
        # ─── UNIVERSAL ORDER LOCK ─────────────────────────────────────────────
        # Programmatically extract all blendshape names in their exact layout order
        # This keeps the Y-axis perfectly uniform across every single expression loop
        universal_blendshape_order = [col for col in df.columns if col != 'Expression']
        
        # Reverse the list so the first column starts at the TOP of the horizontal bar chart
        universal_blendshape_order.reverse()
        
        # Loop over every single expression in the cohort matrix
        for expr in EXPRESSIONS:
            expr_filtered = df[df['Expression'] == expr]
            
            if expr_filtered.empty:
                print(f"   ⚠️ Row not found for: '{expr}'. Skipping.")
                continue
                
            # Isolate muscle weights and map them explicitly to our locked master order
            muscle_series = expr_filtered.iloc[0].drop('Expression').astype(float)
            ordered_muscles = muscle_series.reindex(universal_blendshape_order)
            
            # ─── PLOT STYLING CONFIGURATION ───────────────────────────────────
            # Expanded height (15 inches) to give all 61 parameters clean personal spacing
            fig, ax = plt.subplots(figsize=(10, 15))
            
            # Draw bars with a clean corporate palette
            ordered_muscles.plot(kind='barh', color='#2c3e50', edgecolor='none', width=0.7, ax=ax)
            
            # CRITICAL: Keep X-axis absolute so value shifts map to spatial shifts
            ax.set_xlim(0.0, 1.0)
            
            # Aesthetic framing adjustments
            display_title = expr.replace('_', ' ')
            clean_group_name = group.replace('_', ' ')
            
            ax.set_title(f"{clean_group_name} Spectrum Profile:\n{display_title}", fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel("Activation Weight Baseline (0.0 to 1.0)", fontsize=11, labelpad=10)
            ax.set_ylabel("Standardized Blendshape Channels (Fixed Order)", fontsize=11, labelpad=10)
            
            # Fine-grain horizontal tracking ticks for easy data auditing
            ax.grid(axis='x', linestyle='--', alpha=0.6, color='#bdc3c7')
            ax.set_axisbelow(True) # Push grid lines behind the bars
            
            # Clear out top/right framing lines for a cleaner modern presentation look
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
                
            plt.tight_layout()
            
            # Save out high-res production files
            save_filename = f"{expr}_equalizer_profile.png"
            full_save_path = os.path.join(output_plot_dir, save_filename)
            
            plt.savefig(full_save_path, dpi=300)
            plt.close()
            
            print(f"   💾 Saved Equalizer: ...\\Muscle Activation Plots\\{save_filename}")
            
        print(f"✅ Finished generating locked spectrum plots for {group}.\n")
        
    print("🎉 Equalizer profiles generated cleanly!")

if __name__ == "__main__":
    batch_generate_equalizer_plots()
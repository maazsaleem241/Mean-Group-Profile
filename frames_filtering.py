import os
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ============================================================
# CONSTANTS & DIRECTORY CONFIGURATION
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

# ============================================================
# ROBUST DIRECTORY & IMAGE FINDING HELPERS
# ============================================================
def get_expression_folder(group, expression, participant):
    """Finds the folder containing the frames for a specific event sequence."""
    variations = [
        expression,
        expression.replace('_', ' '),
        expression.replace('_', '-')
    ]
    if expression == 'Tension_Stress':
        variations.extend(['Tension/Stress', 'Tension_Stress', 'Tension Stress'])
        
    for var in variations:
        path = os.path.join(BASE_DIR, "Groups", group, "expressions", var, participant)
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, "Groups", group, "expressions", expression, participant)

def find_event_image(folder_path, event_id, frame_index):
    """Locates the correct frame image file by event ID or frame index."""
    if not os.path.exists(folder_path):
        return None
    files = os.listdir(folder_path)
    
    for f in files:
        if event_id.lower() in f.lower():
            return os.path.join(folder_path, f)
            
    for f in files:
        if str(frame_index) in f:
            return os.path.join(folder_path, f)
            
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder_path, f)
    return None

# ============================================================
# INTERACTIVE OUTLIER SCRUBBING INTERFACE
# ============================================================
class OutlierScrubberGUI:
    def __init__(self, root, target_group, selected_participant=None, selected_expression=None):
        self.root = root
        self.target_group = target_group
        
        all_participants = sorted(GROUP_MAPPING[target_group], key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
        
        if selected_participant and selected_participant != "All Participants":
            self.participants = [selected_participant]
        else:
            self.participants = all_participants
            
        if selected_expression and selected_expression != "All Expressions":
            try:
                start_idx = EXPRESSIONS.index(selected_expression)
                self.expression_scope = EXPRESSIONS[start_idx:]
            except ValueError:
                self.expression_scope = EXPRESSIONS
        else:
            self.expression_scope = EXPRESSIONS
            
        self.root.title(f"Blendshape Outlier Scrubbing Pipeline - {target_group}")
        self.root.geometry("950x700")
        self.root.configure(bg="#2c3e50")
        
        self.p_idx = 0       
        self.expr_idx = 0    
        self.ev_idx = 0      
        
        self.full_participant_df = None
        self.filtered_events_list = []
        self.outliers_set = self.load_outliers_log()
        
        self.setup_ui()
        self.bind_keys()
        self.load_participant_csv()

    def load_outliers_log(self):
        """Loads outliers into a set of unique (Participant, Event_ID) tuples to protect cross-user logs."""
        if os.path.exists(OUTLIER_LOG_FILE):
            try:
                df = pd.read_csv(OUTLIER_LOG_FILE)
                df_outliers = df[df['Status'] == 'Outlier']
                return set(zip(df_outliers['Participant'].astype(str), df_outliers['Event_ID'].astype(str)))
            except Exception:
                return set()
        return set()

    def save_outlier_status(self, event_id, is_outlier):
        p_id = self.participants[self.p_idx]
        composite_key = (str(p_id), str(event_id))
        status_str = "Outlier" if is_outlier else "Valid"
        
        if is_outlier:
            self.outliers_set.add(composite_key)
        else:
            self.outliers_set.discard(composite_key)
            
        if os.path.exists(OUTLIER_LOG_FILE):
            df = pd.read_csv(OUTLIER_LOG_FILE)
            # Safely drop old record for this specific participant + event pair
            df = df[~((df['Participant'] == p_id) & (df['Event_ID'] == event_id))]
        else:
            df = pd.DataFrame(columns=['Group', 'Participant', 'Event_ID', 'Status'])
            
        new_row = pd.DataFrame([{
            'Group': self.target_group,
            'Participant': p_id,
            'Event_ID': event_id,
            'Status': status_str
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(OUTLIER_LOG_FILE, index=False)

    def setup_ui(self):
        self.meta_frame = tk.Frame(self.root, bg="#34495e", height=60)
        self.meta_frame.pack(fill=tk.X)
        
        self.lbl_p_info = tk.Label(self.meta_frame, text="Participant: --", font=("Arial", 12, "bold"), fg="#ecf0f1", bg="#34495e")
        self.lbl_p_info.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.lbl_counter = tk.Label(self.meta_frame, text="Event Check: --/--", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        self.lbl_counter.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.display_frame = tk.Frame(self.root, bg="#1a252f")
        self.display_frame.pack(expand=True, fill=tk.BOTH, padx=25, pady=15)
        
        self.lbl_status_banner = tk.Label(self.display_frame, text="VALID EVENT", font=("Arial", 14, "bold"), fg="#2ecc71", bg="#1a252f")
        self.lbl_status_banner.pack(pady=5)
        
        self.img_label = tk.Label(self.display_frame, bg="#1a252f")
        self.img_label.pack(expand=True, fill=tk.BOTH, pady=5)
        
        self.lbl_details = tk.Label(self.display_frame, text="Details: --", font=("Arial", 11), fg="#ecf0f1", bg="#1a252f")
        self.lbl_details.pack(pady=5)
        
        self.help_frame = tk.Frame(self.root, bg="#2c3e50")
        self.help_frame.pack(fill=tk.X, pady=10)
        self.lbl_help = tk.Label(self.help_frame, text="Controls:  -> / Space: KEEP VALID  |  [Delete] / [Backspace]: FLAG AS OUTLIER  |  <- Arrow: PREV  |  [Q]: Exit", font=("Arial", 11, "italic"), fg="#ecf0f1", bg="#2c3e50")
        self.lbl_help.pack()

    def bind_keys(self):
        self.root.bind("<Right>", lambda e: self.move_next())
        self.root.bind("<space>", lambda e: self.move_next())
        self.root.bind("<Left>", lambda e: self.move_prev())
        self.root.bind("<Delete>", lambda e: self.flag_outlier())
        self.root.bind("<BackSpace>", lambda e: self.flag_outlier())
        self.root.bind("<q>", lambda e: self.root.destroy())
        self.root.bind("<Q>", lambda e: self.root.destroy())

    def load_participant_csv(self):
        if self.p_idx >= len(self.participants):
            messagebox.showinfo("Pipeline Complete", "Selected curation profile evaluation finished!")
            self.root.destroy()
            return
            
        p_id = self.participants[self.p_idx]
        evidence_path = os.path.join(BASE_DIR, "Groups", self.target_group, "data", f"{p_id}_event_evidence.csv")
            
        if not os.path.exists(evidence_path):
            print(f"⚠️ Missing event evidence for {p_id} at: {evidence_path}, skipping participant...")
            self.p_idx += 1
            self.load_participant_csv()
            return
            
        try:
            self.full_participant_df = pd.read_csv(evidence_path)
            self.expr_idx = 0
            self.prepare_expression_slice()
        except Exception as e:
            print(f"Error loading evidence file for {p_id}: {e}")
            self.p_idx += 1
            self.load_participant_csv()

    def prepare_expression_slice(self):
        if self.expr_idx >= len(self.expression_scope):
            self.p_idx += 1
            self.load_participant_csv()
            return

        current_expr = self.expression_scope[self.expr_idx]
        expr_csv_names = [current_expr, current_expr.replace('_', ' ')]
        if current_expr == 'Tension_Stress':
            expr_csv_names.extend(['Tension/Stress', 'Tension Stress'])

        df_filtered = self.full_participant_df[self.full_participant_df['Expression'].isin(expr_csv_names)]
        self.filtered_events_list = df_filtered.to_dict('records')[:100]
        self.ev_idx = 0
        
        if not self.filtered_events_list:
            self.expr_idx += 1
            self.prepare_expression_slice()
            return
            
        self.display_event()

    def display_event(self):
        p_id = self.participants[self.p_idx]
        current_expr = self.expression_scope[self.expr_idx]
        ev = self.filtered_events_list[self.ev_idx]
        
        event_id = ev['Event ID']
        frame_idx = ev['Frame Index']
        timestamp = ev['Timestamp']
        peak_int = ev['Peak Intensity']
        
        clean_expr = current_expr.replace(' ', '_').replace('/', '_')
        
        self.lbl_p_info.config(text=f"Participant: {p_id} ({self.p_idx+1}/{len(self.participants)})  |  Expression: {current_expr.upper()}")
        self.lbl_counter.config(text=f"Expression Frame: {self.ev_idx + 1} / {len(self.filtered_events_list)} (Capped at 100)")
        self.lbl_details.config(text=f"ID: {event_id}  |  Frame Row Index: {frame_idx}  |  Timestamp: {timestamp}  |  Intensity: {peak_int:.3f}")
        
        composite_key = (str(p_id), str(event_id))
        if composite_key in self.outliers_set:
            self.lbl_status_banner.config(text="🚨 EXTREME OUTLIER (FLAGGED TO BE IGNORED)", fg="#e74c3c")
            self.display_frame.config(bg="#3c1f1f")
            self.img_label.config(bg="#3c1f1f")
        else:
            self.lbl_status_banner.config(text="✅ VALID OCCURRENCE (WILL BE COMPILED)", fg="#2ecc71")
            self.display_frame.config(bg="#1a252f")
            self.img_label.config(bg="#1a252f")
            
        folder_path = get_expression_folder(self.target_group, clean_expr, p_id)
        img_path = find_event_image(folder_path, event_id, frame_idx)
        
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((700, 430))
                self.tk_img = ImageTk.PhotoImage(img)
                self.img_label.config(image=self.tk_img, text="")
            except Exception:
                self.img_label.config(image="", text="⚠️ Error Loading Image Frame File", fg="#f1c40f")
        else:
            self.img_label.config(image="", text=f"🖼️ No Image Frame Cached in Folder:\n{folder_path}\n(Data will still be scrubbed correctly based on row index)", fg="#bdc3c7")

    def move_next(self):
        if self.ev_idx < len(self.filtered_events_list) - 1:
            self.ev_idx += 1
            self.display_event()
        else:
            p_id = self.participants[self.p_idx]
            current_expr = self.expression_scope[self.expr_idx]
            
            messagebox.showinfo("Expression Complete", f"Those were all the frames for '{current_expr}' for {p_id}.\nMoving on to next segment layer...")
            
            self.root.deiconify()
            self.root.update()
            self.root.focus_force()
            
            self.expr_idx += 1
            self.prepare_expression_slice()

    def move_prev(self):
        if self.ev_idx > 0:
            self.ev_idx -= 1
            self.display_event()
        elif self.expr_idx > 0:
            self.expr_idx -= 1
            current_expr = self.expression_scope[self.expr_idx]
            expr_csv_names = [current_expr, current_expr.replace('_', ' ')]
            if current_expr == 'Tension_Stress':
                expr_csv_names.extend(['Tension/Stress', 'Tension Stress'])
                
            df_filtered = self.full_participant_df[self.full_participant_df['Expression'].isin(expr_csv_names)]
            self.filtered_events_list = df_filtered.to_dict('records')[:100]
            self.ev_idx = len(self.filtered_events_list) - 1 if self.filtered_events_list else 0
            if self.filtered_events_list:
                self.display_event()

    def flag_outlier(self):
        current_event_id = self.filtered_events_list[self.ev_idx]['Event ID']
        p_id = self.participants[self.p_idx]
        composite_key = (str(p_id), str(current_event_id))
        
        if composite_key in self.outliers_set:
            self.save_outlier_status(current_event_id, is_outlier=False)
            print(f"🔄 Restored Event ID: {current_event_id} to Valid status.")
        else:
            self.save_outlier_status(current_event_id, is_outlier=True)
            print(f"❌ Flagged Event ID: {current_event_id} as an Outlier.")
        self.display_event()

# ============================================================
# LAUNCH PANEL CENTRAL DIALOG BOX WITH TARGET FILTERS
# ============================================================
class PipelineLauncherWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Hierarchical Analysis Entry Matrix")
        self.master.geometry("480x320")
        self.master.configure(bg="#34495e")
        
        lbl_welcome = tk.Label(master, text="Two-Tier Blendshape Curation Filter", font=("Arial", 12, "bold"), fg="#ecf0f1", bg="#34495e")
        lbl_welcome.pack(pady=(20, 10))
        
        lbl_prompt = tk.Label(master, text="Select Targeted Cohort Group:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        lbl_prompt.pack(pady=5)
        
        self.group_var = tk.StringVar()
        self.combo_group = ttk.Combobox(master, textvariable=self.group_var, values=list(GROUP_MAPPING.keys()), state="readonly", font=("Arial", 11), width=26)
        self.combo_group.pack(pady=5)
        self.combo_group.bind("<<ComboboxSelected>>", self.update_participant_options)
        
        lbl_p_prompt = tk.Label(master, text="Select Target Participant:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        lbl_p_prompt.pack(pady=5)
        
        self.p_var = tk.StringVar()
        self.combo_participant = ttk.Combobox(master, textvariable=self.p_var, state="readonly", font=("Arial", 11), width=26)
        self.combo_participant.pack(pady=5)
        
        lbl_expr_prompt = tk.Label(master, text="Resume/Start from Expression Phase:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        lbl_expr_prompt.pack(pady=5)
        
        self.expr_var = tk.StringVar()
        self.combo_expression = ttk.Combobox(master, textvariable=self.expr_var, values=["All Expressions"] + EXPRESSIONS, state="readonly", font=("Arial", 11), width=26)
        self.combo_expression.pack(pady=5)
        self.combo_expression.current(0)
        
        self.combo_group.current(0)
        self.update_participant_options(None)
        
        btn_gui = tk.Button(master, text="LAUNCH OUTLIER FILTERING GUI", font=("Arial", 10, "bold"), bg="#3498db", fg="white", width=34, height=2, command=self.open_scrubber)
        btn_gui.pack(pady=(20, 5))

    def update_participant_options(self, event):
        selected_group = self.group_var.get()
        raw_list = GROUP_MAPPING[selected_group]
        sorted_list = sorted(raw_list, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
        
        self.combo_participant['values'] = ["All Participants"] + sorted_list
        self.combo_participant.current(0)

    def open_scrubber(self):
        selected_group = self.group_var.get()
        chosen_participant = self.p_var.get()
        chosen_expression = self.expr_var.get()
        gui_root = tk.Toplevel(self.master)
        OutlierScrubberGUI(gui_root, selected_group, chosen_participant, chosen_expression)

if __name__ == "__main__":
    launcher_root = tk.Tk()
    app = PipelineLauncherWindow(launcher_root)
    launcher_root.mainloop()

    
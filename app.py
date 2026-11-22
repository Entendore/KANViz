import sys
import ast
import time
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QComboBox, QLineEdit, QTextEdit, QFormLayout, 
                               QSpinBox, QProgressBar, QGroupBox, QDoubleSpinBox,
                               QTabWidget, QTableView, QHeaderView, QRadioButton,
                               QListWidget, QListWidgetItem, QMessageBox, QScrollArea,
                               QCheckBox, QStatusBar, QSplitter, QGridLayout)
from PySide6.QtCore import QThread, Signal, Qt, QAbstractTableModel, QMimeData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, confusion_matrix, classification_report
from kan import KAN

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- Custom Table Models ---
class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None): return self._df.shape[0]
    def columnCount(self, parent=None): return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole: return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal: return str(self._df.columns[section])
            elif orientation == Qt.Vertical: return str(self._df.index[section])
        return None

class TrainingHistoryModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._data = []

    def rowCount(self, parent=None): return len(self._data)
    def columnCount(self, parent=None): return 3

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return ["Step", "Train Loss", "Test Loss"][section]
        return None

    def add_row(self, step, train_loss, test_loss):
        self.beginInsertRows(self.index(len(self._data), 0), len(self._data), len(self._data))
        self._data.append([step, f"{train_loss:.6f}", f"{test_loss:.6f}"])
        self.endInsertRows()

    def clear_data(self):
        self.beginResetModel()
        self._data = []
        self.endResetModel()

# --- Training Thread ---
class TrainThread(QThread):
    progress = Signal(int, int)
    metric_update = Signal(int, float, float)
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(self, train_loader, X_test, y_test, width, grid, k, steps, lr, task_type, patience, device, scheduler_type, affine, noise_scale, optimizer_type, weight_decay, grad_clip):
        super().__init__()
        self.train_loader = train_loader
        self.X_test = X_test.to(device)
        self.y_test = y_test.to(device)
        self.width = width
        self.grid = grid
        self.k = k
        self.steps = steps
        self.lr = lr
        self.task_type = task_type
        self.patience = patience
        self.device = device
        self.scheduler_type = scheduler_type
        self.affine = affine
        self.noise_scale = noise_scale
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.stop_flag = False

    def stop(self): self.stop_flag = True

    def run(self):
        try:
            model = KAN(width=self.width, grid=self.grid, k=self.k, affine_trainable=self.affine, noise_scale=self.noise_scale)
            model = model.to(self.device)
            
            if self.optimizer_type == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_type == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                
            if self.scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            elif self.scheduler_type == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
            else:
                scheduler = None
                
            loss_fn = torch.nn.CrossEntropyLoss() if self.task_type == "classification" else torch.nn.MSELoss()
            best_metric = float('inf')
            best_state_dict = None
            patience_counter = 0
            step = 0
            
            while step < self.steps and not self.stop_flag:
                for batch_x, batch_y in self.train_loader:
                    if self.stop_flag or step >= self.steps: break
                    
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    model.train()
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                        
                    optimizer.step()
                    if scheduler: scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        test_pred = model(self.X_test)
                        test_metric = loss_fn(test_pred, self.y_test).item()

                    self.metric_update.emit(step + 1, loss.item(), test_metric)
                    self.progress.emit(step + 1, self.steps)
                    
                    if test_metric < best_metric:
                        best_metric = test_metric
                        best_state_dict = model.state_dict() # Save best weights
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.patience > 0 and patience_counter >= self.patience:
                            self.finished.emit(best_state_dict, f"Early Stopping at step {step+1}! Best Loss: {best_metric:.4f}")
                            return

                    if step % max(1, self.steps // 5) == 0:
                        try: model.update_grid(batch_x)
                        except: pass 
                    step += 1

            msg = f"Training aborted at step {step}." if self.stop_flag else "Training Complete!"
            self.finished.emit(best_state_dict if best_state_dict is not None else model.state_dict(), msg)
        except Exception as e:
            self.error.emit(str(e))

# --- Matplotlib Canvas ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, n_subplots=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if n_subplots == 1:
            self.ax = self.fig.add_subplot(111)
            self.ax2 = None
        else:
            self.ax = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
        super().__init__(self.fig)
        self.init_plot()

    def init_plot(self, metric_name="Loss"):
        self.ax.clear()
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel(metric_name)
        self.fig.tight_layout()
        self.draw()

    def update_plot(self, steps, train_metrics, test_metrics, metric_name, snapshot=None):
        self.ax.clear()
        self.ax.plot(steps, train_metrics, label=f"Train {metric_name}", color='blue')
        self.ax.plot(steps, test_metrics, label=f"Test {metric_name}", color='red')
        
        if snapshot:
            s_steps, s_train, s_test = snapshot
            self.ax.plot(s_steps, s_train, label="Snap Train", color='blue', linestyle='--', alpha=0.5)
            self.ax.plot(s_steps, s_test, label="Snap Test", color='red', linestyle='--', alpha=0.5)
            
        self.ax.set_title("Training Metrics (Live)")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel(metric_name)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.draw()

# --- Main Application ---
class KANApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KAN Studio v10: Ultimate MLOps Workstation")
        self.resize(1300, 880)
        self.setAcceptDrops(True)
        
        self.df = None
        self.worker = None
        self.trained_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.label_encoder = None
        self.ordinal_encoder = None
        self.feature_columns = []
        self.categorical_mappings = {}
        self.current_task_type = "regression"
        
        self.plot_steps, self.plot_train, self.plot_test = [], [], []
        self.snapshot_data = None
        
        self.X_test_np, self.y_test_np = None, None
        self.X_train_full = None # Stored for refined symbolic fitting
        self.y_train_full = None
        
        self.dark_mode = True
        self.history_model = TrainingHistoryModel()
        
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= LEFT PANEL =================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # --- Top Bar ---
        top_bar = QHBoxLayout()
        self.btn_theme = QPushButton("🌙 Dark Mode")
        self.btn_theme.setCheckable(True)
        self.btn_theme.setChecked(True)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_clear_logs = QPushButton("🗑️ Clear Logs")
        self.btn_clear_logs.clicked.connect(self.log_window.clear if hasattr(self, 'log_window') else lambda: None)
        self.btn_reset_config = QPushButton("↩️ Reset Defaults")
        self.btn_reset_config.clicked.connect(self.reset_config_defaults)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_reset_config)
        top_bar.addWidget(self.btn_theme)
        top_bar.addWidget(self.btn_clear_logs)
        left_layout.addLayout(top_bar)
        
        # --- Project Group ---
        proj_group = QGroupBox("Project")
        proj_layout = QHBoxLayout()
        self.btn_load_proj = QPushButton("📂 Load")
        self.btn_load_proj.clicked.connect(self.load_project)
        self.btn_save_proj = QPushButton("💾 Save")
        self.btn_save_proj.setEnabled(False)
        self.btn_save_proj.clicked.connect(self.save_project)
        proj_layout.addWidget(self.btn_load_proj)
        proj_layout.addWidget(self.btn_save_proj)
        proj_group.setLayout(proj_layout)
        left_layout.addWidget(proj_group)
        
        # --- Data Group ---
        data_group = QGroupBox("1. Dataset")
        data_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load CSV Dataset (or Drag & Drop)")
        self.btn_load.clicked.connect(self.load_dataset)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        
        form_data = QFormLayout()
        self.combo_target = QComboBox()
        self.combo_target.setEnabled(False)
        self.combo_target.currentTextChanged.connect(self.on_target_changed)
        form_data.addRow("Target:", self.combo_target)
        
        data_split_form = QFormLayout()
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.1, 0.5); self.spin_test_size.setSingleStep(0.05); self.spin_test_size.setValue(0.2)
        data_split_form.addRow("Test Size:", self.spin_test_size)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999); self.spin_seed.setValue(42)
        data_split_form.addRow("Seed:", self.spin_seed)
        
        data_layout.addWidget(self.btn_load)
        data_layout.addWidget(self.lbl_file)
        data_layout.addLayout(form_data)
        data_layout.addLayout(data_split_form)
        data_layout.addWidget(QLabel("Features:"))
        self.list_features = QListWidget()
        self.list_features.setMaximumHeight(80)
        data_layout.addWidget(self.list_features)
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        # --- Task & Hardware ---
        sys_group = QGroupBox("2. Task & Hardware")
        sys_layout = QVBoxLayout()
        task_layout = QHBoxLayout()
        self.radio_reg = QRadioButton("Regression")
        self.radio_cls = QRadioButton("Classification")
        self.radio_reg.setChecked(True)
        self.radio_reg.toggled.connect(self.on_target_changed)
        task_layout.addWidget(self.radio_reg)
        task_layout.addWidget(self.radio_cls)
        sys_layout.addLayout(task_layout)
        
        hw_layout = QHBoxLayout()
        self.check_gpu = QCheckBox("Use GPU (CUDA)")
        if not torch.cuda.is_available(): 
            self.check_gpu.setEnabled(False)
            self.check_gpu.setToolTip("CUDA not detected on this system.")
        hw_layout.addWidget(self.check_gpu)
        sys_layout.addLayout(hw_layout)
        sys_group.setLayout(sys_layout)
        left_layout.addWidget(sys_group)

        # --- Model Group ---
        config_group = QGroupBox("3. KAN Configuration")
        config_layout = QFormLayout()
        
        self.input_width = QLineEdit("[4, 3, 1]")
        self.input_width.setToolTip("Network architecture. e.g. [inputs, hidden, outputs]")
        config_layout.addRow("Width:", self.input_width)
        
        self.spin_grid = QSpinBox(); self.spin_grid.setMinimum(1); self.spin_grid.setMaximum(20); self.spin_grid.setValue(5)
        self.spin_grid.setToolTip("Number of grid points for splines. Higher = more complex functions.")
        config_layout.addRow("Grid:", self.spin_grid)
        
        self.spin_k = QSpinBox(); self.spin_k.setMinimum(1); self.spin_k.setMaximum(5); self.spin_k.setValue(3)
        self.spin_k.setToolTip("Order of the spline polynomial.")
        config_layout.addRow("K (Spline):", self.spin_k)
        
        self.check_affine = QCheckBox("Affine Trainable")
        self.check_affine.setChecked(True)
        self.check_affine.setToolTip("Allows scaling and shifting of splines. Usually improves accuracy.")
        config_layout.addRow("Advanced:", self.check_affine)
        
        self.spin_noise = QDoubleSpinBox(); self.spin_noise.setRange(0.0, 1.0); self.spin_noise.setValue(0.1); self.spin_noise.setSingleStep(0.05)
        self.spin_noise.setToolTip("Adds noise to splines during training to prevent overfitting.")
        config_layout.addRow("Noise Scale:", self.spin_noise)
        
        self.spin_steps = QSpinBox(); self.spin_steps.setMinimum(10); self.spin_steps.setMaximum(10000); self.spin_steps.setValue(200)
        config_layout.addRow("Max Steps:", self.spin_steps)
        
        self.spin_batch = QSpinBox(); self.spin_batch.setMinimum(4); self.spin_batch.setMaximum(1024); self.spin_batch.setValue(32)
        config_layout.addRow("Batch Size:", self.spin_batch)
        
        self.spin_patience = QSpinBox(); self.spin_patience.setMinimum(0); self.spin_patience.setMaximum(1000); self.spin_patience.setValue(20)
        self.spin_patience.setToolTip("Stops training if test loss doesn't improve for N steps. 0 = disabled.")
        config_layout.addRow("Patience:", self.spin_patience)
        
        self.input_lr = QDoubleSpinBox(); self.input_lr.setValue(0.02); self.input_lr.setSingleStep(0.01); self.input_lr.setDecimals(4)
        config_layout.addRow("Learn Rate:", self.input_lr)
        
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setRange(0.0, 1.0); self.spin_wd.setValue(0.0); self.spin_wd.setSingleStep(0.001); self.spin_wd.setDecimals(4)
        self.spin_wd.setToolTip("L2 Regularization to prevent overfitting.")
        config_layout.addRow("Weight Decay:", self.spin_wd)
        
        self.spin_grad_clip = QDoubleSpinBox(); self.spin_grad_clip.setRange(0.0, 10.0); self.spin_grad_clip.setValue(0.0); self.spin_grad_clip.setSingleStep(0.1)
        self.spin_grad_clip.setToolTip("Prevents exploding gradients. 0 = disabled.")
        config_layout.addRow("Grad Clip:", self.spin_grad_clip)
        
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["Adam", "AdamW", "SGD"])
        config_layout.addRow("Optimizer:", self.combo_optimizer)
        
        self.combo_scheduler = QComboBox()
        self.combo_scheduler.addItems(["None", "StepLR", "CosineAnnealingLR"])
        config_layout.addRow("Scheduler:", self.combo_scheduler)
        
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # --- Symbolic Library Config ---
        sym_group = QGroupBox("4. Symbolic Library")
        sym_layout = QGridLayout()
        self.sym_checks = {}
        syms = ['x', 'x^2', 'x^3', 'x^4', '1/x', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs']
        for i, s in enumerate(syms):
            chk = QCheckBox(s)
            if s in ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']:
                chk.setChecked(True)
            sym_layout.addWidget(chk, i//3, i%3)
            self.sym_checks[s] = chk
        sym_group.setLayout(sym_layout)
        left_layout.addWidget(sym_group)

        # --- Action Group ---
        action_group = QGroupBox("5. Execution & Analysis")
        action_layout = QVBoxLayout()
        
        self.btn_train = QPushButton("🚀 Train KAN")
        self.btn_train.setEnabled(False); self.btn_train.setMinimumHeight(40)
        self.btn_train.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_train.clicked.connect(self.start_training)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white;")
        self.btn_stop.clicked.connect(self.stop_training)
        
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        
        btn_layout0 = QHBoxLayout()
        self.btn_snapshot = QPushButton("📸 Snapshot Run")
        self.btn_snapshot.setToolTip("Saves current loss curves. Next run will overlay this snapshot for A/B comparison.")
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        btn_layout0.addWidget(self.btn_snapshot)
        action_layout.addLayout(btn_layout0)
        
        btn_layout1 = QHBoxLayout()
        self.btn_summary = QPushButton("📝 Summary"); self.btn_summary.clicked.connect(self.show_model_summary)
        self.btn_evaluate = QPushButton("📊 Evaluate"); self.btn_evaluate.setEnabled(False); self.btn_evaluate.clicked.connect(self.evaluate_model)
        btn_layout1.addWidget(self.btn_summary); btn_layout1.addWidget(self.btn_evaluate)
        
        btn_layout2 = QHBoxLayout()
        self.btn_feat_imp = QPushButton("🔍 Perm. Imp."); self.btn_feat_imp.setEnabled(False); self.btn_feat_imp.clicked.connect(self.calculate_permutation_importance)
        self.btn_native_imp = QPushButton("🧠 Native Imp."); self.btn_native_imp.setEnabled(False); self.btn_native_imp.clicked.connect(self.calculate_native_importance)
        btn_layout2.addWidget(self.btn_feat_imp); btn_layout2.addWidget(self.btn_native_imp)
        
        btn_layout3 = QHBoxLayout()
        self.btn_prune_plot = QPushButton("✂️ Prune & Plot"); self.btn_prune_plot.setEnabled(False); self.btn_prune_plot.clicked.connect(self.prune_and_plot)
        self.btn_formula = QPushButton("🧮 Formula"); self.btn_formula.setEnabled(False); self.btn_formula.clicked.connect(self.extract_formula)
        btn_layout3.addWidget(self.btn_prune_plot); btn_layout3.addWidget(self.btn_formula)
        
        btn_layout4 = QHBoxLayout()
        self.btn_refine_formula = QPushButton("🔬 Refine Formula"); self.btn_refine_formula.setEnabled(False); self.btn_refine_formula.clicked.connect(self.refine_symbolic_formula)
        btn_layout4.addWidget(self.btn_refine_formula)

        action_layout.addWidget(self.btn_train)
        action_layout.addWidget(self.btn_stop)
        action_layout.addWidget(self.progress_bar)
        action_layout.addLayout(btn_layout1)
        action_layout.addLayout(btn_layout2)
        action_layout.addLayout(btn_layout3)
        action_layout.addLayout(btn_layout4)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)

        # --- Log Window ---
        left_layout.addWidget(QLabel("Console:"))
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True); self.log_window.setMaximumHeight(100)
        left_layout.addWidget(self.log_window)
        left_layout.addStretch()
        
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll, stretch=1)

        # ================= RIGHT PANEL (Tabs) =================
        self.tabs = QTabWidget()
        
        # Data Explorer (Table + Correlation + Dist + Stats)
        data_tab_widget = QSplitter(Qt.Vertical)
        
        eda_tabs = QTabWidget()
        self.table_view = QTableView()
        eda_tabs.addTab(self.table_view, "Data Preview")
        
        self.stats_view = QTableView()
        eda_tabs.addTab(self.stats_view, "Statistics")
        
        self.missing_view = QTableView()
        eda_tabs.addTab(self.missing_view, "Missing Values")
        
        data_tab_widget.addWidget(eda_tabs)
        
        plots_splitter = QSplitter(Qt.Horizontal)
        corr_group = QGroupBox("Correlation Matrix")
        corr_layout = QVBoxLayout()
        self.corr_canvas = MplCanvas(self, width=5, height=4, dpi=100, n_subplots=1)
        corr_layout.addWidget(self.corr_canvas)
        corr_group.setLayout(corr_layout)
        plots_splitter.addWidget(corr_group)
        
        dist_group = QGroupBox("Target Distribution")
        dist_layout = QVBoxLayout()
        self.dist_canvas = MplCanvas(self, width=5, height=4, dpi=100, n_subplots=1)
        dist_layout.addWidget(self.dist_canvas)
        dist_group.setLayout(dist_layout)
        plots_splitter.addWidget(dist_group)
        
        data_tab_widget.addWidget(plots_splitter)
        self.tabs.addTab(data_tab_widget, "Data Explorer")
        
        # Training Metrics Split
        train_tab_widget = QSplitter(Qt.Vertical)
        self.canvas = MplCanvas(self, width=6, height=4, dpi=100, n_subplots=1)
        train_tab_widget.addWidget(self.canvas)
        
        history_group = QGroupBox("Training History Log")
        history_layout = QVBoxLayout()
        self.history_table = QTableView()
        self.history_table.setModel(self.history_model)
        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        train_tab_widget.addWidget(history_group)
        self.tabs.addTab(train_tab_widget, "Training Metrics")
        
        # Evaluation Tab
        self.eval_widget = QWidget()
        eval_layout = QVBoxLayout(self.eval_widget)
        eval_top_bar = QHBoxLayout()
        self.eval_metrics_label = QLabel("Train model and click 'Evaluate' to see metrics.")
        self.eval_metrics_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_export_eval = QPushButton("💾 Export Eval CSV")
        self.btn_export_eval.setEnabled(False)
        self.btn_export_eval.clicked.connect(self.export_evaluation)
        eval_top_bar.addWidget(self.eval_metrics_label)
        eval_top_bar.addStretch()
        eval_top_bar.addWidget(self.btn_export_eval)
        eval_layout.addLayout(eval_top_bar)
        
        self.eval_canvas = MplCanvas(self.eval_widget, width=8, height=5, dpi=100, n_subplots=2)
        eval_layout.addWidget(self.eval_canvas)
        
        self.eval_report_text = QTextEdit()
        self.eval_report_text.setReadOnly(True)
        self.eval_report_text.setMaximumHeight(150)
        eval_layout.addWidget(self.eval_report_text)
        self.tabs.addTab(self.eval_widget, "Evaluation & Residuals")
        
        # Feature Importance Tab
        self.imp_canvas = MplCanvas(self, width=6, height=5, dpi=100, n_subplots=1)
        self.tabs.addTab(self.imp_canvas, "Feature Importance")
        
        self.kan_plot_widget = QWidget()
        self.kan_plot_layout = QVBoxLayout(self.kan_plot_widget)
        self.tabs.addTab(self.kan_plot_widget, "KAN Architecture")
        
        self.formula_text = QTextEdit()
        self.formula_text.setReadOnly(True)
        self.tabs.addTab(self.formula_text, "Symbolic Formula")
        
        # Inference Tab
        self.predict_widget = QWidget()
        predict_layout = QVBoxLayout(self.predict_widget)
        infer_btn_layout = QHBoxLayout()
        self.btn_load_pred_csv = QPushButton("📂 Load CSV for Batch Predict")
        self.btn_load_pred_csv.setEnabled(False)
        self.btn_load_pred_csv.clicked.connect(self.load_batch_predict_csv)
        self.btn_batch_predict = QPushButton("⚡ Run Batch Prediction & Export")
        self.btn_batch_predict.setEnabled(False)
        self.btn_batch_predict.clicked.connect(self.run_batch_prediction)
        infer_btn_layout.addWidget(self.btn_load_pred_csv)
        infer_btn_layout.addWidget(self.btn_batch_predict)
        predict_layout.addLayout(infer_btn_layout)
        
        predict_layout.addWidget(QLabel("Manual Single Inference:"))
        self.predict_form_container = QWidget()
        self.predict_form_layout = QFormLayout(self.predict_form_container)
        predict_layout.addWidget(self.predict_form_container)
        
        self.btn_predict = QPushButton("🔮 Predict Single Row")
        self.btn_predict.setEnabled(False); self.btn_predict.clicked.connect(self.run_prediction)
        predict_layout.addWidget(self.btn_predict)
        
        self.lbl_prediction_result = QLabel("Prediction will appear here")
        self.lbl_prediction_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        self.lbl_prediction_result.setWordWrap(True)
        predict_layout.addWidget(self.lbl_prediction_result)
        predict_layout.addStretch()
        self.tabs.addTab(self.predict_widget, "Inference")
        
        main_layout.addWidget(self.tabs, stretch=2)
        
        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def reset_config_defaults(self):
        self.input_width.setText("[4, 3, 1]")
        self.spin_grid.setValue(5)
        self.spin_k.setValue(3)
        self.check_affine.setChecked(True)
        self.spin_noise.setValue(0.1)
        self.spin_steps.setValue(200)
        self.spin_batch.setValue(32)
        self.spin_patience.setValue(20)
        self.input_lr.setValue(0.02)
        self.spin_wd.setValue(0.0)
        self.spin_grad_clip.setValue(0.0)
        self.combo_optimizer.setCurrentIndex(0)
        self.combo_scheduler.setCurrentIndex(0)
        self.log("↩️ Configuration reset to defaults.")

    # --- Drag & Drop Overrides ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.csv'):
                self.load_dataset(file_path)
                break

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.btn_theme.setText("🌙 Dark Mode" if self.dark_mode else "☀️ Light Mode")
        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            app.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; }
                QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 15px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
                QPushButton { background-color: #333; color: white; border: 1px solid #555; padding: 6px; border-radius: 3px; }
                QPushButton:hover { background-color: #444; }
                QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableView { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #555; border-radius: 2px; }
                QHeaderView::section { background-color: #333; padding: 4px; border: 1px solid #555; font-weight: bold; }
                QListWidget { background-color: #2d2d2d; color: #d4d4d4; }
                QTabWidget::pane { border: 1px solid #444; }
                QSplitter::handle { background-color: #444; }
            """)
        else:
            app.setStyleSheet("""
                QMainWindow { background-color: #f5f5f5; }
                QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
                QPushButton { padding: 6px; }
                QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; }
                QTableView { gridline-color: #ddd; background-color: white; }
                QHeaderView::section { background-color: #eee; padding: 4px; border: 1px solid #ccc; font-weight: bold; }
                QListWidget { background-color: white; }
            """)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_window.append(f"[{timestamp}] {message}")

    def load_dataset(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.lbl_file.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.status_bar.showMessage(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} cols")
                
                model = PandasModel(self.df.head(50))
                self.table_view.setModel(model)
                self.table_view.resizeColumnsToContents()
                
                # Stats & Missing Values
                stats_df = self.df.describe().transpose()
                self.stats_view.setModel(PandasModel(stats_df))
                self.stats_view.resizeColumnsToContents()
                
                missing_df = pd.DataFrame({
                    'Missing Count': self.df.isnull().sum(),
                    'Missing %': (self.df.isnull().sum() / len(self.df)) * 100
                })
                self.missing_view.setModel(PandasModel(missing_df))
                self.missing_view.resizeColumnsToContents()
                
                self.combo_target.clear()
                self.combo_target.addItems(self.df.columns)
                self.combo_target.setEnabled(True)
                self.btn_train.setEnabled(True)
                
                self.list_features.clear()
                for col in self.df.columns:
                    item = QListWidgetItem(col)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    self.list_features.addItem(item)
                self.list_features.itemChanged.connect(self.on_target_changed)
                
                self.plot_correlation_matrix()
                self.reset_results()
                self.log(f"✅ Loaded dataset: {self.df.shape}")
                self.on_target_changed()
            except Exception as e:
                self.log(f"❌ Error: {str(e)}")

    def plot_correlation_matrix(self):
        if self.df is None: return
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2: return
        
        corr = numeric_df.corr()
        self.corr_canvas.ax.clear()
        im = self.corr_canvas.ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        self.corr_canvas.ax.set_xticks(np.arange(len(corr.columns)))
        self.corr_canvas.ax.set_yticks(np.arange(len(corr.columns)))
        self.corr_canvas.ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        self.corr_canvas.ax.set_yticklabels(corr.columns)
        
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                self.corr_canvas.ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
                
        self.corr_canvas.ax.set_title("Pearson Correlation Matrix")
        self.corr_canvas.fig.tight_layout()
        self.corr_canvas.draw()

    def plot_target_distribution(self):
        if self.df is None or self.combo_target.count() == 0: return
        target = self.combo_target.currentText()
        self.dist_canvas.ax.clear()
        
        if self.radio_reg.isChecked():
            self.dist_canvas.ax.hist(self.df[target].dropna(), bins=30, color='steelblue', edgecolor='black')
            self.dist_canvas.ax.set_title(f"Distribution of {target}")
            self.dist_canvas.ax.set_xlabel(target)
            self.dist_canvas.ax.set_ylabel("Frequency")
        else:
            counts = self.df[target].value_counts()
            self.dist_canvas.ax.bar(counts.index.astype(str), counts.values, color='coral')
            self.dist_canvas.ax.set_title(f"Class Balance: {target}")
            self.dist_canvas.ax.set_xlabel("Class")
            self.dist_canvas.ax.set_ylabel("Count")
            self.dist_canvas.ax.tick_params(axis='x', rotation=45)
            
        self.dist_canvas.fig.tight_layout()
        self.dist_canvas.draw()

    def on_target_changed(self):
        if self.df is None or self.combo_target.count() == 0: return
        target = self.combo_target.currentText()
        checked_features = []
        for i in range(self.list_features.count()):
            item = self.list_features.item(i)
            if item.text() != target and item.checkState() == Qt.Checked:
                checked_features.append(item.text())
                
        num_features = len(checked_features)
        if self.radio_reg.isChecked():
            out_size = 1
            self.canvas.init_plot("Loss (MSE)")
            self.current_task_type = "regression"
        else:
            out_size = self.df[target].nunique()
            self.canvas.init_plot("Loss (CrossEntropy)")
            self.current_task_type = "classification"
            
        self.input_width.setText(f"[{num_features}, {2*num_features+1}, {out_size}]")
        self.plot_target_distribution()

    def reset_results(self):
        self.btn_save_proj.setEnabled(False)
        self.btn_prune_plot.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_load_pred_csv.setEnabled(False)
        self.btn_batch_predict.setEnabled(False)
        self.btn_evaluate.setEnabled(False)
        self.btn_export_eval.setEnabled(False)
        self.btn_formula.setEnabled(False)
        self.btn_feat_imp.setEnabled(False)
        self.btn_native_imp.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.btn_refine_formula.setEnabled(False)
        self.progress_bar.setValue(0)
        self.plot_steps, self.plot_train, self.plot_test = [], [], []
        self.history_model.clear_data()
        self.lbl_prediction_result.setText("Prediction will appear here")
        self.eval_metrics_label.setText("Train model and click 'Evaluate' to see metrics.")
        self.eval_report_text.clear()
        self.eval_canvas.ax.clear()
        self.eval_canvas.ax2.clear()
        self.eval_canvas.draw()
        self.imp_canvas.ax.clear()
        self.imp_canvas.draw()
        self.formula_text.clear()
        
        while self.kan_plot_layout.count():
            item = self.kan_plot_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def toggle_buttons_during_training(self, is_training):
        self.btn_train.setEnabled(not is_training)
        self.btn_stop.setEnabled(is_training)
        self.btn_evaluate.setEnabled(not is_training and self.trained_model is not None)
        self.btn_formula.setEnabled(not is_training and self.trained_model is not None)
        self.btn_prune_plot.setEnabled(not is_training and self.trained_model is not None)
        self.btn_feat_imp.setEnabled(not is_training and self.trained_model is not None)
        self.btn_native_imp.setEnabled(not is_training and self.trained_model is not None)
        self.btn_predict.setEnabled(not is_training and self.trained_model is not None)
        self.btn_refine_formula.setEnabled(not is_training and self.trained_model is not None)

    def show_model_summary(self):
        try:
            width = ast.literal_eval(self.input_width.text())
            temp_model = KAN(width=width, grid=self.spin_grid.value(), k=self.spin_k.value())
            num_params = sum(p.numel() for p in temp_model.parameters())
            mem_mb = num_params * 4 / (1024 ** 2)
            
            summary = f"KAN Architecture Summary:\n{'-'*30}\n"
            summary += f"Width: {width}\nGrid: {self.spin_grid.value()}\nK: {self.spin_k.value()}\n"
            summary += f"Total Parameters: {num_params:,}\n"
            summary += f"Estimated Memory: {mem_mb:.4f} MB\n{'-'*30}\n"
            summary += str(temp_model)
            
            QMessageBox.information(self, "Model Summary", summary)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid configuration:\n{str(e)}")

    def start_training(self):
        if self.df is None: return
        try:
            target_col = self.combo_target.currentText()
            width = ast.literal_eval(self.input_width.text())
            grid = self.spin_grid.value(); k = self.spin_k.value()
            steps = self.spin_steps.value(); batch_size = self.spin_batch.value()
            patience = self.spin_patience.value(); lr = self.input_lr.value()
            test_size = self.spin_test_size.value(); seed = self.spin_seed.value()
            task_type = "regression" if self.radio_reg.isChecked() else "classification"
            device = "cuda" if (self.check_gpu.isChecked() and torch.cuda.is_available()) else "cpu"
            scheduler_type = self.combo_scheduler.currentText()
            affine = self.check_affine.isChecked()
            noise_scale = self.spin_noise.value()
            optimizer_type = self.combo_optimizer.currentText()
            weight_decay = self.spin_wd.value()
            grad_clip = self.spin_grad_clip.value()

            self.feature_columns = []
            for i in range(self.list_features.count()):
                item = self.list_features.item(i)
                if item.text() != target_col and item.checkState() == Qt.Checked:
                    self.feature_columns.append(item.text())

            if len(self.feature_columns) == 0: raise ValueError("No features selected!")

            df_clean = self.df.dropna()
            
            cat_cols = df_clean[self.feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = [c for c in self.feature_columns if c not in cat_cols]
            
            X_num = df_clean[num_cols].values.astype(np.float32) if num_cols else np.empty((len(df_clean), 0), dtype=np.float32)
            self.categorical_mappings = {}
            
            if cat_cols:
                self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_cat = self.ordinal_encoder.fit_transform(df_clean[cat_cols]).astype(np.float32)
                X = np.hstack((X_num, X_cat))
                for i, col in enumerate(cat_cols):
                    cats = self.ordinal_encoder.categories_[i]
                    self.categorical_mappings[col] = list(cats)
            else:
                self.ordinal_encoder = None
                X = X_num

            if task_type == "regression":
                y = df_clean[target_col].values.reshape(-1, 1).astype(np.float32)
                self.label_encoder = None
            else:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(df_clean[target_col])

            if width[0] != X.shape[1] or width[-1] != (1 if task_type == "regression" else len(np.unique(y))):
                self.log(f"❌ Width mismatch! Expected in={X.shape[1]}, out={1 if task_type == 'regression' else len(np.unique(y))}.")
                return

            self.log(f"⚙️ Preprocessing data... [Device: {device.upper()}]")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
            
            self.scaler_X = StandardScaler()
            X_train = self.scaler_X.fit_transform(X_train).astype(np.float32)
            X_test = self.scaler_X.transform(X_test).astype(np.float32)

            if task_type == "regression":
                self.scaler_y = StandardScaler()
                y_train = self.scaler_y.fit_transform(y_train).astype(np.float32)
                y_test = self.scaler_y.transform(y_test).astype(np.float32)
            else: self.scaler_y = None

            self.X_test_np = X_test
            self.y_test_np = y_test
            self.X_train_full = X_train
            self.y_train_full = y_train

            tensor_x = torch.Tensor(X_train)
            tensor_y = torch.Tensor(y_train).long() if task_type == "classification" else torch.Tensor(y_train)
            dataset = TensorDataset(tensor_x, tensor_y)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            X_test_t = torch.Tensor(X_test)
            y_test_t = torch.Tensor(y_test).long() if task_type == "classification" else torch.Tensor(y_test)

            self.reset_results()
            self.toggle_buttons_during_training(True)
            self.tabs.setCurrentIndex(1)
            self.log(f"🚀 Starting training...")

            self.worker = TrainThread(train_loader, X_test_t, y_test_t, width, grid, k, steps, lr, task_type, patience, device, scheduler_type, affine, noise_scale, optimizer_type, weight_decay, grad_clip)
            self.worker.progress.connect(self.update_progress)
            self.worker.metric_update.connect(self.update_metrics)
            self.worker.finished.connect(self.on_training_finished)
            self.worker.error.connect(self.on_training_error)
            self.worker.start()

        except Exception as e:
            self.log(f"❌ Setup Error: {str(e)}")
            self.toggle_buttons_during_training(False)

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("⏹️ Stop requested...")

    def update_progress(self, current, total):
        pct = int((current / total) * 100)
        self.progress_bar.setValue(pct)
        self.status_bar.showMessage(f"Training... {pct}%")

    def update_metrics(self, step, train_metric, test_metric):
        self.plot_steps.append(step)
        self.plot_train.append(train_metric)
        self.plot_test.append(test_metric)
        self.history_model.add_row(step, train_metric, test_metric)
        
        if step % 5 == 0 or step == 1:
            metric_name = "Loss (MSE)" if self.current_task_type == "regression" else "Loss (CE)"
            self.canvas.update_plot(self.plot_steps, self.plot_train, self.plot_test, metric_name, self.snapshot_data)

    def take_snapshot(self):
        if len(self.plot_steps) > 0:
            self.snapshot_data = (list(self.plot_steps), list(self.plot_train), list(self.plot_test))
            self.log("📸 Snapshot taken. Next training run will be compared against this.")
            self.btn_snapshot.setEnabled(False)

    def on_training_finished(self, state_dict, message):
        self.log(f"✅ {message}")
        width = ast.literal_eval(self.input_width.text())
        affine = self.check_affine.isChecked()
        noise = self.spin_noise.value()
        
        self.trained_model = KAN(width=width, grid=self.spin_grid.value(), k=self.spin_k.value(), affine_trainable=affine, noise_scale=noise)
        self.trained_model.load_state_dict(state_dict)
        
        self.toggle_buttons_during_training(False)
        self.btn_save_proj.setEnabled(True)
        self.btn_load_pred_csv.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        self.setup_prediction_tab()
        self.status_bar.showMessage("Training complete. Ready for analysis.")

    def on_training_error(self, message):
        self.log(f"❌ Training Error: {message}")
        self.toggle_buttons_during_training(False)
        self.progress_bar.setValue(0)

    def setup_prediction_tab(self):
        while self.predict_form_layout.count():
            item = self.predict_form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.predict_inputs = {}
        for col in self.feature_columns:
            if col in self.categorical_mappings:
                inp = QComboBox()
                inp.addItems([str(x) for x in self.categorical_mappings[col]])
            else:
                inp = QDoubleSpinBox()
                inp.setRange(-9999999.0, 9999999.0)
            self.predict_form_layout.addRow(f"{col}:", inp)
            self.predict_inputs[col] = inp

    def run_prediction(self):
        if not self.trained_model: return
        try:
            input_dict = {}
            for col, widget in self.predict_inputs.items():
                if isinstance(widget, QComboBox):
                    input_dict[col] = widget.currentText()
                else:
                    input_dict[col] = widget.value()
                    
            X_raw_df = pd.DataFrame([input_dict])
            X_raw_df = X_raw_df[self.feature_columns]
            
            cat_cols = [c for c in self.feature_columns if c in self.categorical_mappings]
            num_cols = [c for c in self.feature_columns if c not in cat_cols]
            
            X_num = X_raw_df[num_cols].values.astype(np.float32) if num_cols else np.empty((1, 0), dtype=np.float32)
            if cat_cols:
                X_cat = self.ordinal_encoder.transform(X_raw_df[cat_cols]).astype(np.float32)
                X_raw = np.hstack((X_num, X_cat))
            else:
                X_raw = X_num
                
            X_scaled = self.scaler_X.transform(X_raw).astype(np.float32)
            
            self.trained_model.eval()
            with torch.no_grad():
                pred_scaled = self.trained_model(torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()
                
            if self.current_task_type == "regression":
                pred_val = self.scaler_y.inverse_transform(pred_scaled)[0][0]
                self.lbl_prediction_result.setText(f"Predicted Value: {pred_val:.4f}")
            else:
                # Apply softmax for probabilities
                probs = torch.softmax(torch.tensor(pred_scaled), dim=1).numpy()[0]
                pred_class_idx = np.argmax(probs)
                pred_class_label = self.label_encoder.inverse_transform([pred_class_idx])[0]
                
                prob_str = "\n".join([f"{self.label_encoder.classes_[i]}: {p:.2%}" for i, p in enumerate(probs)])
                self.lbl_prediction_result.setText(f"Predicted Class: {pred_class_label}\n\nProbabilities:\n{prob_str}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def load_batch_predict_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV for Prediction", "", "CSV Files (*.csv)")
        if path:
            self.batch_df = pd.read_csv(path)
            missing_cols = [c for c in self.feature_columns if c not in self.batch_df.columns]
            if missing_cols:
                QMessageBox.critical(self, "Missing Columns", f"The uploaded CSV is missing required feature columns: {missing_cols}")
                self.batch_df = None
                return
            self.btn_batch_predict.setEnabled(True)
            self.log(f"📂 Loaded batch CSV with shape: {self.batch_df.shape}")

    def run_batch_prediction(self):
        if not hasattr(self, 'batch_df') or self.batch_df is None: return
        
        X_raw_df = self.batch_df[self.feature_columns]
        cat_cols = [c for c in self.feature_columns if c in self.categorical_mappings]
        num_cols = [c for c in self.feature_columns if c not in cat_cols]
        
        X_num = X_raw_df[num_cols].values.astype(np.float32) if num_cols else np.empty((len(X_raw_df), 0), dtype=np.float32)
        if cat_cols:
            X_cat = self.ordinal_encoder.transform(X_raw_df[cat_cols]).astype(np.float32)
            X = np.hstack((X_num, X_cat))
        else:
            X = X_num
            
        X_scaled = self.scaler_X.transform(X).astype(np.float32)
        
        self.trained_model.eval()
        with torch.no_grad():
            preds_scaled = self.trained_model(torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()
            
        if self.current_task_type == "regression":
            preds = self.scaler_y.inverse_transform(preds_scaled).flatten()
            self.batch_df['Prediction'] = preds
        else:
            pred_idxs = np.argmax(preds_scaled, axis=1)
            preds = self.label_encoder.inverse_transform(pred_idxs)
            self.batch_df['Prediction'] = preds
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "predictions.csv", "CSV Files (*.csv)")
        if save_path:
            self.batch_df.to_csv(save_path, index=False)
            self.log(f"✅ Batch predictions exported to {save_path}")
            QMessageBox.information(self, "Success", f"Predictions saved successfully!")

    def evaluate_model(self):
        if not self.trained_model or self.X_test_np is None: return
        self.log("📊 Evaluating model on test set...")
        self.tabs.setCurrentIndex(2)
        
        self.trained_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(self.X_test_np, dtype=torch.float32)
            preds_scaled = self.trained_model(X_t).cpu().numpy()
            
        if self.current_task_type == "regression":
            y_true = self.scaler_y.inverse_transform(self.y_test_np)
            y_pred = self.scaler_y.inverse_transform(preds_scaled)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            self.eval_metrics_label.setText(f"Test R² Score: {r2:.4f}  |  Test RMSE: {rmse:.4f}")
            
            self.eval_y_true = y_true.flatten()
            self.eval_y_pred = y_pred.flatten()
            
            self.eval_canvas.ax.clear()
            self.eval_canvas.ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
            self.eval_canvas.ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            self.eval_canvas.ax.set_title("True vs Predicted")
            self.eval_canvas.ax.set_xlabel("True Values")
            self.eval_canvas.ax.set_ylabel("Predicted Values")
            self.eval_canvas.ax.grid(True, linestyle='--', alpha=0.6)
            
            residuals = y_true.flatten() - y_pred.flatten()
            self.eval_canvas.ax2.clear()
            self.eval_canvas.ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
            self.eval_canvas.ax2.axhline(y=0, color='r', linestyle='--')
            self.eval_canvas.ax2.set_title("Residual Plot")
            self.eval_canvas.ax2.set_xlabel("Predicted Values")
            self.eval_canvas.ax2.set_ylabel("Residuals")
            self.eval_canvas.ax2.grid(True, linestyle='--', alpha=0.6)
            
            self.eval_canvas.fig.tight_layout()
            self.eval_canvas.draw()
            self.eval_report_text.clear()
        else:
            y_true = self.y_test_np
            y_pred = np.argmax(preds_scaled, axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            self.eval_metrics_label.setText(f"Test Accuracy: {acc:.4f}  |  Test F1 Score: {f1:.4f}")
            
            self.eval_y_true = y_true
            self.eval_y_pred = y_pred
            
            self.eval_canvas.ax.clear()
            cm = confusion_matrix(y_true, y_pred)
            self.eval_canvas.ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            self.eval_canvas.ax.set_title("Confusion Matrix")
            self.eval_canvas.ax.set_ylabel("True label")
            self.eval_canvas.ax.set_xlabel("Predicted label")
            
            self.eval_canvas.ax2.clear()
            self.eval_canvas.ax2.axis('off')
            self.eval_canvas.fig.tight_layout()
            self.eval_canvas.draw()
            
            report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_.astype(str))
            self.eval_report_text.setPlainText(report)
            
        self.btn_export_eval.setEnabled(True)
        self.log("✅ Evaluation complete.")

    def export_evaluation(self):
        if not hasattr(self, 'eval_y_true'): return
        path, _ = QFileDialog.getSaveFileName(self, "Save Evaluation Results", "eval_results.csv", "CSV Files (*.csv)")
        if path:
            df_eval = pd.DataFrame({'True_Value': self.eval_y_true, 'Predicted_Value': self.eval_y_pred})
            if self.current_task_type == "regression":
                df_eval['Residual'] = df_eval['True_Value'] - df_eval['Predicted_Value']
            df_eval.to_csv(path, index=False)
            self.log(f"💾 Evaluation results exported to {path}")

    def calculate_permutation_importance(self):
        if not self.trained_model or self.X_test_np is None: return
        self.log("🔍 Calculating Permutation Feature Importance...")
        self.tabs.setCurrentIndex(3)
        
        self.trained_model.eval()
        X_t = torch.tensor(self.X_test_np, dtype=torch.float32)
        
        with torch.no_grad():
            if self.current_task_type == "regression":
                base_loss = torch.nn.MSELoss()(self.trained_model(X_t), torch.tensor(self.y_test_np, dtype=torch.float32)).item()
            else:
                base_loss = torch.nn.CrossEntropyLoss()(self.trained_model(X_t), torch.tensor(self.y_test_np, dtype=torch.long)).item()
                
        importances = []
        for i in range(X_t.shape[1]):
            X_permuted = X_t.clone()
            X_permuted[:, i] = X_permuted[torch.randperm(X_t.shape[0]), i]
            
            with torch.no_grad():
                if self.current_task_type == "regression":
                    perm_loss = torch.nn.MSELoss()(self.trained_model(X_permuted), torch.tensor(self.y_test_np, dtype=torch.float32)).item()
                else:
                    perm_loss = torch.nn.CrossEntropyLoss()(self.trained_model(X_permuted), torch.tensor(self.y_test_np, dtype=torch.long)).item()
                    
            importances.append(perm_loss - base_loss)
            
        self.imp_canvas.ax.clear()
        sorted_idx = np.argsort(importances)
        self.imp_canvas.ax.barh(range(len(sorted_idx)), [importances[i] for i in sorted_idx], color='purple', alpha=0.7)
        self.imp_canvas.ax.set_yticks(range(len(sorted_idx)))
        self.imp_canvas.ax.set_yticklabels([self.feature_columns[i] for i in sorted_idx])
        self.imp_canvas.ax.set_title("Permutation Feature Importance")
        self.imp_canvas.ax.set_xlabel("Increase in Loss (Importance)")
        self.imp_canvas.fig.tight_layout()
        self.imp_canvas.draw()
        self.log("✅ Permutation importance calculated.")

    def calculate_native_importance(self):
        if not self.trained_model: return
        self.log("🧠 Calculating Native KAN Feature Importance (L1 Norm)...")
        self.tabs.setCurrentIndex(3)
        try:
            imp_df = self.trained_model.feature()
            
            self.imp_canvas.ax.clear()
            scores = imp_df.iloc[:, 0].values.flatten()
            labels = imp_df.index.tolist()
            
            sorted_idx = np.argsort(scores)
            self.imp_canvas.ax.barh(range(len(sorted_idx)), [scores[i] for i in sorted_idx], color='darkorange', alpha=0.7)
            self.imp_canvas.ax.set_yticks(range(len(sorted_idx)))
            self.imp_canvas.ax.set_yticklabels([labels[i] for i in sorted_idx])
            self.imp_canvas.ax.set_title("Native KAN Feature Importance (L1)")
            self.imp_canvas.ax.set_xlabel("L1 Edge Weight Score")
            self.imp_canvas.fig.tight_layout()
            self.imp_canvas.draw()
            self.log("✅ Native importance calculated.")
        except Exception as e:
            self.log(f"❌ Native Importance Error: {str(e)}")

    def extract_formula(self):
        if not self.trained_model: return
        try:
            self.log("🧮 Extracting symbolic formula...")
            lib = [k for k, v in self.sym_checks.items() if v.isChecked()]
            if len(lib) == 0:
                QMessageBox.warning(self, "Library Empty", "Please select at least one symbolic function.")
                return
                
            self.trained_model.auto_symbolic(lib=lib)
            formula = self.trained_model.symbolic_formula(floating_digits=4)
            
            text = ""
            for i, f in enumerate(formula):
                text += f"Output {i+1}:\n{str(f)}\n\n"
            self.formula_text.setPlainText(text)
            self.tabs.setCurrentIndex(5)
            self.log("✅ Formula extracted. Click 'Refine Formula' to fit precise coefficients.")
        except Exception as e:
            self.log(f"❌ Formula Error: {str(e)}")

    def refine_symbolic_formula(self):
        if not self.trained_model: return
        try:
            self.log("🔬 Refining symbolic formula (Fitting coefficients)...")
            self.btn_refine_formula.setEnabled(False)
            
            X_t = torch.tensor(self.X_train_full, dtype=torch.float32)
            y_t = torch.tensor(self.y_train_full, dtype=torch.float32 if self.current_task_type == "regression" else torch.long)
            
            # Run a few steps of regression to fit the affine parameters of the symbolic model
            optimizer = torch.optim.LBFGS(self.trained_model.parameters(), lr=1.0)
            
            def closure():
                optimizer.zero_grad()
                pred = self.trained_model(X_t)
                if self.current_task_type == "regression":
                    loss = torch.nn.MSELoss()(pred, y_t)
                else:
                    loss = torch.nn.CrossEntropyLoss()(pred, y_t)
                loss.backward()
                return loss
                
            for _ in range(5):
                optimizer.step(closure)
                
            formula = self.trained_model.symbolic_formula(floating_digits=4)
            text = ""
            for i, f in enumerate(formula):
                text += f"Output {i+1} (Refined):\n{str(f)}\n\n"
            self.formula_text.setPlainText(text)
            self.tabs.setCurrentIndex(5)
            self.log("✅ Symbolic formula refined with precise coefficients.")
        except Exception as e:
            self.log(f"❌ Refinement Error: {str(e)}")
        finally:
            self.btn_refine_formula.setEnabled(True)

    def save_project(self):
        if not self.trained_model: return
        path, _ = QFileDialog.getSaveFileName(self, "Save KAN Project", "project.kanp", "KAN Project (*.kanp)")
        if path:
            try:
                project_data = {
                    'model_state': self.trained_model.state_dict(),
                    'width': ast.literal_eval(self.input_width.text()),
                    'grid': self.spin_grid.value(),
                    'k': self.spin_k.value(),
                    'task_type': self.current_task_type,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'label_encoder': self.label_encoder,
                    'ordinal_encoder': self.ordinal_encoder,
                    'categorical_mappings': self.categorical_mappings,
                    'feature_columns': self.feature_columns,
                    'affine': self.check_affine.isChecked(),
                    'noise': self.spin_noise.value()
                }
                with open(path, 'wb') as f: pickle.dump(project_data, f)
                self.log(f"💾 Project saved to {path}")
            except Exception as e:
                self.log(f"❌ Error saving project: {str(e)}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load KAN Project", "", "KAN Project (*.kanp)")
        if path:
            try:
                with open(path, 'rb') as f: project_data = pickle.load(f)
                    
                self.check_affine.setChecked(project_data.get('affine', False))
                self.spin_noise.setValue(project_data.get('noise', 0.1))
                
                self.trained_model = KAN(width=project_data['width'], grid=project_data['grid'], k=project_data['k'], 
                                         affine_trainable=project_data.get('affine', False), noise_scale=project_data.get('noise', 0.1))
                self.trained_model.load_state_dict(project_data['model_state'])
                
                self.scaler_X = project_data['scaler_X']
                self.scaler_y = project_data['scaler_y']
                self.label_encoder = project_data['label_encoder']
                self.ordinal_encoder = project_data.get('ordinal_encoder')
                self.categorical_mappings = project_data.get('categorical_mappings', {})
                self.feature_columns = project_data['feature_columns']
                self.current_task_type = project_data['task_type']
                
                self.input_width.setText(str(project_data['width']))
                self.spin_grid.setValue(project_data['grid'])
                self.spin_k.setValue(project_data['k'])
                
                if self.current_task_type == "classification": self.radio_cls.setChecked(True)
                else: self.radio_reg.setChecked(True)
                    
                self.btn_save_proj.setEnabled(True); self.btn_prune_plot.setEnabled(True)
                self.btn_formula.setEnabled(True); self.btn_load_pred_csv.setEnabled(True)
                self.btn_evaluate.setEnabled(False)
                self.btn_feat_imp.setEnabled(False)
                self.btn_native_imp.setEnabled(True)
                self.btn_refine_formula.setEnabled(False) # Needs training data to refine
                self.setup_prediction_tab()
                self.log(f"📂 Project loaded. Ready for inference.")
            except Exception as e:
                self.log(f"❌ Error loading project: {str(e)}")

    def prune_and_plot(self):
        if self.trained_model:
            try:
                self.log("✂️ Pruning model...")
                self.trained_model = self.trained_model.cpu()
                pruned_model = self.trained_model.prune()
                
                while self.kan_plot_layout.count():
                    item = self.kan_plot_layout.takeAt(0)
                    if item.widget(): item.widget().deleteLater()
                
                plt.close('all')
                pruned_model.plot()
                fig = plt.gcf()
                
                canvas = FigureCanvasQTAgg(fig)
                self.kan_plot_layout.addWidget(canvas)
                self.tabs.setCurrentIndex(4)
                self.log("🖼️ KAN architecture plot generated.")
            except Exception as e:
                self.log(f"❌ Plot Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KANApp()
    window.show()
    sys.exit(app.exec())
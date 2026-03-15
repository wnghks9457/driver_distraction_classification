import os
import re
import gc
import random
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)

from xgboost import XGBClassifier


# =========================================================
# USER CONFIGURATION
# =========================================================
class Config:
    FOLDER_PATH = "Distraction_dataset_Final_Merged"
    RESULTS_DIR = "Results_XGBoost_4Class_FileBasedSplit"

    SEED = 42

    TARGET_LABELS_MAP = {
        0: 0,   # ND
        1: 1,   # CD
        2: 2,   # ED
        3: 3    # MD
    }
    CLASS_NAMES = ['ND', 'CD', 'ED', 'MD']

    FPS = 28
    WINDOW_SECONDS = 10
    STRIDE_SECONDS = 5
    TIME_STEPS = FPS * WINDOW_SECONDS

    STRICT_WINDOW_LABEL = False

    N_SPLITS = 5
    USE_SCALER = True

    SAVE_FOLD_CM = True
    SAVE_FOLD_ROC = False
    SAVE_COMBINED_CM = True
    SAVE_COMBINED_ROC = True

    MODES = [
        'AU', 'Pose', 'Vehicle', 'Gaze',
        'AU+Pose', 'AU+Vehicle', 'AU+Gaze',
        'Pose+Vehicle', 'Pose+Gaze', 'Vehicle+Gaze',
        'AU+Pose+Vehicle+Gaze'
    ]

    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': SEED,
        'tree_method': 'hist',
        'n_jobs': -1
    }


# =========================================================
# UTILS
# =========================================================
def set_seeds(seed=Config.SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] 랜덤 시드 설정: {seed}")


def write_log(filepath, message, print_console=True):
    if print_console:
        print(message)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(message + "\n")


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)


def format_class_distribution(y_array, class_names):
    unique, counts = np.unique(y_array, return_counts=True)
    return {class_names[int(k)]: int(v) for k, v in zip(unique, counts)}


def pad_or_truncate_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if seq.shape[0] < target_len:
        pad_width = target_len - seq.shape[0]
        if seq.shape[0] == 0:
            return np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        return np.pad(seq, ((0, pad_width), (0, 0)), mode='edge').astype(np.float32)
    elif seq.shape[0] > target_len:
        return seq[:target_len].astype(np.float32)
    return seq.astype(np.float32)


def get_feature_groups():
    au_features = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
        'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
        'AU25_r', 'AU26_r', 'AU45_r'
    ]
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
    gaze_raw_features = [
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y'
    ]
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']
    gaze_total_features = gaze_raw_features + kinematic_features

    return au_features, pose_features, vehicle_features, gaze_raw_features, gaze_total_features


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_subject_data(df, au_features, gaze_angle_features, pose_features):
    df = df.copy()
    sampling_rate = Config.FPS

    if 'timestamp' in df.columns and len(df) > 1:
        dt = df['timestamp'].diff().replace(0, np.nan)
        mean_dt = dt.mean()
        if pd.isna(mean_dt) or mean_dt <= 0:
            mean_dt = 1.0 / sampling_rate
        dt = dt.fillna(mean_dt).astype(np.float32)
    else:
        dt = pd.Series(np.full(len(df), 1.0 / sampling_rate, dtype=np.float32))

    if all(col in df.columns for col in ['gaze_angle_x', 'gaze_angle_y']):
        dx = df['gaze_angle_x'].diff().fillna(0)
        dy = df['gaze_angle_y'].diff().fillna(0)

        amp_rad = np.sqrt(dx**2 + dy**2).astype(np.float32)
        df['gaze_amp'] = np.degrees(amp_rad).astype(np.float32)

        gaze_vel = (df['gaze_amp'] / dt).replace([np.inf, -np.inf], 0).fillna(0)
        df['gaze_vel'] = gaze_vel.astype(np.float32)

        d_vel = df['gaze_vel'].diff().fillna(0)
        gaze_acc = (d_vel / dt).replace([np.inf, -np.inf], 0).fillna(0)
        df['gaze_acc'] = gaze_acc.astype(np.float32)

        if len(df) >= 2:
            df.loc[df.index[:2], 'gaze_acc'] = 0.0
    else:
        df['gaze_amp'] = 0.0
        df['gaze_vel'] = 0.0
        df['gaze_acc'] = 0.0

    nd_data = df[df['Distraction'] == 0]
    if not nd_data.empty:
        target_cols = [c for c in (au_features + gaze_angle_features + pose_features) if c in df.columns]
        if target_cols:
            baseline_mean = nd_data[target_cols].mean()
            df[target_cols] = df[target_cols] - baseline_mean

    return df


# =========================================================
# DYNAMIC DATA LOADING (FOLD-ISOLATED)
# =========================================================
def create_windows_from_file_list(file_list):
    target_time_steps = Config.TIME_STEPS
    au_features, pose_features, vehicle_features, gaze_raw_features, gaze_total_features = get_feature_groups()
    columns_to_load = au_features + pose_features + vehicle_features + gaze_raw_features + ['Distraction']
    final_features = au_features + pose_features + vehicle_features + gaze_total_features

    X_au_list, X_pose_list, X_vehicle_list, X_gaze_list = [], [], [], []
    y_list = []

    for file in file_list:
        try:
            header_cols = pd.read_csv(file, nrows=0).columns.tolist()
            usecols = [c for c in columns_to_load + ['timestamp'] if c in header_cols]
            df = pd.read_csv(file, usecols=usecols)

            for col in columns_to_load:
                if col not in df.columns:
                    df[col] = 0.0

            df.fillna(0, inplace=True)

            if 'timestamp' in df.columns and len(df) > 1:
                duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                actual_fps = ((len(df) - 1) / duration) if duration > 0 else Config.FPS
                if not np.isfinite(actual_fps) or actual_fps <= 0:
                    actual_fps = Config.FPS
            else:
                actual_fps = Config.FPS

            file_window_size = max(int(round(actual_fps * Config.WINDOW_SECONDS)), 1)
            file_step_size = max(int(round(actual_fps * Config.STRIDE_SECONDS)), 1)

            df = preprocess_subject_data(
                df,
                au_features=au_features,
                gaze_angle_features=['gaze_angle_x', 'gaze_angle_y'],
                pose_features=pose_features
            )

            for col in final_features:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = df[col].astype(np.float32)

            labels = df['Distraction'].values

            if len(df) < file_window_size:
                continue

            for i in range(0, len(df) - file_window_size + 1, file_step_size):
                window_df = df.iloc[i:i + file_window_size]
                window_labels = labels[i:i + file_window_size]

                if Config.STRICT_WINDOW_LABEL and len(np.unique(window_labels)) != 1:
                    continue

                raw_label = int(window_labels[-1])
                if raw_label not in Config.TARGET_LABELS_MAP:
                    continue

                final_label = Config.TARGET_LABELS_MAP[raw_label]

                X_au_list.append(pad_or_truncate_sequence(window_df[au_features].values, target_time_steps))
                X_pose_list.append(pad_or_truncate_sequence(window_df[pose_features].values, target_time_steps))
                X_vehicle_list.append(pad_or_truncate_sequence(window_df[vehicle_features].values, target_time_steps))
                X_gaze_list.append(pad_or_truncate_sequence(window_df[gaze_total_features].values, target_time_steps))
                y_list.append(final_label)

        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({file}): {e}")

    if not X_au_list:
        return None, None, None, None, None

    return (
        np.array(X_au_list, dtype=np.float32),
        np.array(X_pose_list, dtype=np.float32),
        np.array(X_vehicle_list, dtype=np.float32),
        np.array(X_gaze_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32)
    )


# =========================================================
# XGBOOST FEATURE ENGINEERING
# =========================================================
def summarize_sequence_features(X_seq: np.ndarray) -> np.ndarray:
    mean_feat = np.mean(X_seq, axis=1)
    std_feat = np.std(X_seq, axis=1)
    min_feat = np.min(X_seq, axis=1)
    max_feat = np.max(X_seq, axis=1)
    median_feat = np.median(X_seq, axis=1)

    first_feat = X_seq[:, 0, :]
    last_feat = X_seq[:, -1, :]
    delta_feat = last_feat - first_feat

    if X_seq.shape[1] > 1:
        diff = np.diff(X_seq, axis=1)
        diff_mean_feat = np.mean(diff, axis=1)
        abs_diff_mean_feat = np.mean(np.abs(diff), axis=1)
    else:
        diff_mean_feat = np.zeros_like(mean_feat)
        abs_diff_mean_feat = np.zeros_like(mean_feat)

    X_static = np.concatenate([
        mean_feat, std_feat, min_feat, max_feat, median_feat,
        last_feat, delta_feat, diff_mean_feat, abs_diff_mean_feat
    ], axis=1).astype(np.float32)

    return X_static


def assemble_mode_features(feature_blocks: dict, mode: str, indices: np.ndarray) -> np.ndarray:
    blocks = mode.split('+')
    return np.hstack([feature_blocks[b][indices] for b in blocks]).astype(np.float32)


# =========================================================
# CLASS BALANCING (Applied inside Fold)
# =========================================================
def balance_classes_by_count(X, y, seed=42, num_classes=4):
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == cls)[0] for cls in range(num_classes)]
    non_empty_sizes = [len(idx) for idx in class_indices if len(idx) > 0]

    if len(non_empty_sizes) <= 1:
        return X, y

    min_count = min(non_empty_sizes)
    selected_indices = []
    
    for cls in range(num_classes):
        cls_idx = class_indices[cls]
        if len(cls_idx) > 0:
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices, dtype=np.int32)
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


# =========================================================
# PLOTTING / EVALUATION
# =========================================================
def plot_multiclass_roc_curve(y_true, y_proba, class_names, save_path, title):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

    valid_curve_count = 0
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        valid_curve_count += 1

    if valid_curve_count == 0:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model_performance(model, X_val, y_val, class_names, fold_no, save_dir, mode_name, log_file):
    num_classes = len(class_names)
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    bacc = balanced_accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)
    rec = recall_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)

    try:
        auc_score = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = 0.0

    cm = confusion_matrix(y_val, y_pred, labels=np.arange(num_classes))
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = TN / (TN + FP)
        class_specificity = np.nan_to_num(class_specificity)

    spec = np.mean(class_specificity)

    log_msg = f"   [{mode_name}] Acc: {acc:.4f}, BAcc: {bacc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    write_log(log_file, log_msg)

    safe_mode_name = sanitize_filename(mode_name)

    if Config.SAVE_FOLD_CM:
        cm_path = os.path.join(save_dir, f"CM_{safe_mode_name}_fold_{fold_no}.png")
        plot_confusion_matrix(cm, class_names, cm_path, f'Fold {fold_no} - {mode_name} CM')

    if Config.SAVE_FOLD_ROC:
        roc_path = os.path.join(save_dir, f"ROC_{safe_mode_name}_fold_{fold_no}.png")
        plot_multiclass_roc_curve(y_val, y_proba, class_names, roc_path, f'Fold {fold_no} - {mode_name} ROC')

    return {'acc': acc, 'bacc': bacc, 'prec': prec, 'rec': rec, 'f1': f1, 'spec': spec, 'auc': auc_score}, y_val, y_pred, y_proba


def plot_kfold_summary(final_results, save_dir, experiment_name):
    modes = list(final_results.keys())
    acc_means = [np.mean(final_results[m]['acc']) for m in modes]
    acc_stds = [np.std(final_results[m]['acc']) for m in modes]
    f1_means = [np.mean(final_results[m]['f1']) for m in modes]
    f1_stds = [np.std(final_results[m]['f1']) for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig_w = max(14, len(modes) * 1.2)
    plt.figure(figsize=(fig_w, 7))

    plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy', capsize=5, color='skyblue', edgecolor='black')
    plt.bar(x + width/2, f1_means, width, yerr=f1_stds, label='Macro F1', capsize=5, color='steelblue', edgecolor='black')

    plt.xlabel('Modes (Feature Combinations)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'K-Fold Summary: {experiment_name}', fontsize=14)
    plt.xticks(x, modes, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(modes)):
        plt.text(x[i] - width/2, acc_means[i] + 0.015, f"{acc_means[i]:.2f}", ha='center', fontsize=9)
        plt.text(x[i] + width/2, f1_means[i] + 0.015, f"{f1_means[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "KFold_Summary_XGBoost_FileBasedSplit.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_seeds()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    LOG_FILE = os.path.join(Config.RESULTS_DIR, "training_result_log_xgboost.txt")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("Experiment Start: XGBoost Multi-Class Classification\n")
        f.write("Split Method    : File-Based StratifiedKFold (Leakage Free)\n")
        f.write("Train Balancing : In-Fold Undersampling\n")
        f.write(f"Classes: {Config.CLASS_NAMES}\n")
        f.write(f"Modes: {Config.MODES}\n")
        f.write("=" * 80 + "\n")

    print("=" * 80)
    print(f"🚀 STARTING XGBOOST EXPERIMENT (FILE-BASED STRATIFIED SPLIT)")
    print("=" * 80)

    # 1. 파일 목록 및 파일 레벨 라벨 추출
    folder_path = Config.FOLDER_PATH
    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        raise SystemExit

    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    file_paths = np.array([os.path.join(folder_path, f) for f in all_files])
    file_labels = []

    for f in all_files:
        if '-005' in f:
            file_labels.append(1) # CD
        elif '-006' in f:
            file_labels.append(2) # ED
        elif '-007' in f:
            file_labels.append(3) # MD
        else:
            file_labels.append(0) # ND

    file_labels = np.array(file_labels)

    metric_names = ['acc', 'bacc', 'prec', 'rec', 'f1', 'spec', 'auc']
    final_results = {mode: {metric: [] for metric in metric_names} for mode in Config.MODES}
    all_predictions = {mode: {'y_true': [], 'y_pred': [], 'y_proba': []} for mode in Config.MODES}

    # 2. 파일 리스트 대상 Stratified K-Fold
    skf_files = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)

    for fold_no, (train_f_idx, val_f_idx) in enumerate(skf_files.split(file_paths, file_labels), 1):
        print("\n" + "-" * 80)
        print(f"Fold {fold_no}/{Config.N_SPLITS} Data Loading & Processing")
        print("-" * 80)

        train_files = file_paths[train_f_idx]
        val_files = file_paths[val_f_idx]

        write_log(LOG_FILE, f"\n--- Fold {fold_no}/{Config.N_SPLITS} ---")
        write_log(LOG_FILE, f"Train Files Count: {len(train_files)} | Val Files Count: {len(val_files)}")

        # 3. 각 세트별로 독립적인 윈도우 생성 (누수 원천 차단)
        print("[INFO] 학습용 파일 윈도우 생성 중...")
        X_tr_au_seq, X_tr_pos_seq, X_tr_veh_seq, X_tr_gaz_seq, y_tr_raw = create_windows_from_file_list(train_files)
        
        print("[INFO] 검증용 파일 윈도우 생성 중...")
        X_val_au_seq, X_val_pos_seq, X_val_veh_seq, X_val_gaz_seq, y_val_raw = create_windows_from_file_list(val_files)

        if X_tr_au_seq is None or X_val_au_seq is None:
            print("[WARNING] 이번 Fold에서 생성된 윈도우가 없습니다. 건너뜁니다.")
            continue

        write_log(LOG_FILE, f"Train Windows: {len(y_tr_raw)} | Val Windows: {len(y_val_raw)}")

        # 4. Feature Engineering (Summary)
        feat_blocks_tr = {
            'AU': summarize_sequence_features(X_tr_au_seq),
            'Pose': summarize_sequence_features(X_tr_pos_seq),
            'Vehicle': summarize_sequence_features(X_tr_veh_seq),
            'Gaze': summarize_sequence_features(X_tr_gaz_seq)
        }
        feat_blocks_val = {
            'AU': summarize_sequence_features(X_val_au_seq),
            'Pose': summarize_sequence_features(X_val_pos_seq),
            'Vehicle': summarize_sequence_features(X_val_veh_seq),
            'Gaze': summarize_sequence_features(X_val_gaz_seq)
        }

        # 메모리 정리
        del X_tr_au_seq, X_tr_pos_seq, X_tr_veh_seq, X_tr_gaz_seq
        del X_val_au_seq, X_val_pos_seq, X_val_veh_seq, X_val_gaz_seq
        gc.collect()

        # 5. Mode별 학습 및 평가
        for mode in Config.MODES:
            mode_save_dir = os.path.join(Config.RESULTS_DIR, sanitize_filename(mode))
            os.makedirs(mode_save_dir, exist_ok=True)
            print(f" > Training Mode: {mode}")

            # 모드별 피처 조립
            X_tr_mode = assemble_mode_features(feat_blocks_tr, mode, np.arange(len(y_tr_raw)))
            X_val_mode = assemble_mode_features(feat_blocks_val, mode, np.arange(len(y_val_raw)))

            # Fold 내부에서 언더샘플링 진행 (분포 맞춤)
            X_tr_bal, y_tr_bal = balance_classes_by_count(X_tr_mode, y_tr_raw, seed=Config.SEED, num_classes=4)
            X_val_bal, y_val_bal = balance_classes_by_count(X_val_mode, y_val_raw, seed=Config.SEED + 1000 + fold_no, num_classes=4)

            if len(y_tr_bal) == 0 or len(y_val_bal) == 0:
                print(f"[WARNING] {mode} 언더샘플링 후 데이터가 없습니다.")
                continue

            # 스케일링
            if Config.USE_SCALER:
                scaler = StandardScaler()
                X_tr_final = scaler.fit_transform(X_tr_bal)
                X_val_final = scaler.transform(X_val_bal)
            else:
                X_tr_final = X_tr_bal
                X_val_final = X_val_bal

            # 모델 학습
            model = XGBClassifier(**Config.XGB_PARAMS)
            model.fit(X_tr_final, y_tr_bal)

            # 평가
            metrics, y_true_fold, y_pred_fold, y_proba_fold = evaluate_model_performance(
                model=model, X_val=X_val_final, y_val=y_val_bal,
                class_names=Config.CLASS_NAMES, fold_no=fold_no,
                save_dir=mode_save_dir, mode_name=mode, log_file=LOG_FILE
            )

            for metric in metric_names:
                final_results[mode][metric].append(metrics[metric])

            all_predictions[mode]['y_true'].extend(y_true_fold.tolist())
            all_predictions[mode]['y_pred'].extend(y_pred_fold.tolist())
            all_predictions[mode]['y_proba'].append(y_proba_fold)

        del feat_blocks_tr, feat_blocks_val
        gc.collect()

    print("\n" + "=" * 80)
    print("📊 FINAL XGBOOST SUMMARY [FILE-BASED STRATIFIED SPLIT]")
    print("=" * 80)
    write_log(LOG_FILE, "\n" + "=" * 80, print_console=False)
    write_log(LOG_FILE, "FINAL XGBOOST SUMMARY [FILE-BASED STRATIFIED SPLIT]", print_console=False)
    write_log(LOG_FILE, "=" * 80, print_console=False)

    for mode in Config.MODES:
        if not final_results[mode]['acc']:
            continue
        summary_lines = [
            f"[{mode}]",
            f"  ACC : {np.mean(final_results[mode]['acc']):.4f} (+/- {np.std(final_results[mode]['acc']):.4f})",
            f"  BACC: {np.mean(final_results[mode]['bacc']):.4f} (+/- {np.std(final_results[mode]['bacc']):.4f})",
            f"  PREC: {np.mean(final_results[mode]['prec']):.4f} (+/- {np.std(final_results[mode]['prec']):.4f})",
            f"  REC : {np.mean(final_results[mode]['rec']):.4f} (+/- {np.std(final_results[mode]['rec']):.4f})",
            f"  F1  : {np.mean(final_results[mode]['f1']):.4f} (+/- {np.std(final_results[mode]['f1']):.4f})",
            f"  SPEC: {np.mean(final_results[mode]['spec']):.4f} (+/- {np.std(final_results[mode]['spec']):.4f})",
            f"  AUC : {np.mean(final_results[mode]['auc']):.4f} (+/- {np.std(final_results[mode]['auc']):.4f})"
        ]
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        write_log(LOG_FILE, summary_text, print_console=False)
        print("-" * 80)

    for mode in Config.MODES:
        if not all_predictions[mode]['y_true']:
            continue
        mode_save_dir = os.path.join(Config.RESULTS_DIR, sanitize_filename(mode))
        y_true_all = np.array(all_predictions[mode]['y_true'], dtype=np.int32)
        y_pred_all = np.array(all_predictions[mode]['y_pred'], dtype=np.int32)
        y_proba_all = np.vstack(all_predictions[mode]['y_proba'])

        if Config.SAVE_COMBINED_CM:
            cm_all = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(Config.CLASS_NAMES)))
            cm_path = os.path.join(mode_save_dir, f"Combined_CM_{sanitize_filename(mode)}.png")
            plot_confusion_matrix(cm_all, Config.CLASS_NAMES, cm_path, f'Total Combined CM - {mode}')

        if Config.SAVE_COMBINED_ROC:
            roc_path = os.path.join(mode_save_dir, f"Combined_ROC_{sanitize_filename(mode)}.png")
            plot_multiclass_roc_curve(y_true_all, y_proba_all, Config.CLASS_NAMES, roc_path, f'Total Combined ROC - {mode}')

    plot_kfold_summary(final_results, Config.RESULTS_DIR, "XGBoost Multi-Class (File-Based Split)")
    print(f"\n[INFO] 모든 결과가 저장되었습니다: {Config.RESULTS_DIR}")
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
    RESULTS_DIR = "Results_XGBoost_4Class_GlobalUndersamplingBeforeKFold"

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

    pose_features = [
        'pose_Tx', 'pose_Ty', 'pose_Tz',
        'pose_Rx', 'pose_Ry', 'pose_Rz'
    ]

    vehicle_features = [
        'Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset'
    ]

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
# DATA LOADING & SLIDING WINDOW
# =========================================================
def load_and_create_sliding_window_data():
    folder_path = Config.FOLDER_PATH
    target_time_steps = Config.TIME_STEPS

    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        return None, None, None, None, None

    csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])
    if not csv_files:
        print(f"[ERROR] CSV 파일이 없습니다: {folder_path}")
        return None, None, None, None, None

    au_features, pose_features, vehicle_features, gaze_raw_features, gaze_total_features = get_feature_groups()

    columns_to_load = au_features + pose_features + vehicle_features + gaze_raw_features + ['Distraction']
    final_features = au_features + pose_features + vehicle_features + gaze_total_features

    print(f"[INFO] 타겟 클래스 매핑: {Config.TARGET_LABELS_MAP}")
    print(f"[INFO] 클래스 이름: {Config.CLASS_NAMES}")
    print(f"[INFO] Feature Order: AU -> Pose -> Vehicle -> Gaze (Total: {len(final_features)})")
    print(f"[INFO] STRICT_WINDOW_LABEL = {Config.STRICT_WINDOW_LABEL}")

    X_au_list, X_pose_list, X_vehicle_list, X_gaze_list = [], [], [], []
    y_list = []

    for file_idx, file in enumerate(csv_files):
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
                print(f"[WARNING] {os.path.basename(file)}: 길이가 윈도우보다 짧아 건너뜁니다.")
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

                X_au = pad_or_truncate_sequence(window_df[au_features].values, target_time_steps)
                X_pose = pad_or_truncate_sequence(window_df[pose_features].values, target_time_steps)
                X_vehicle = pad_or_truncate_sequence(window_df[vehicle_features].values, target_time_steps)
                X_gaze = pad_or_truncate_sequence(window_df[gaze_total_features].values, target_time_steps)

                X_au_list.append(X_au)
                X_pose_list.append(X_pose)
                X_vehicle_list.append(X_vehicle)
                X_gaze_list.append(X_gaze)
                y_list.append(final_label)

        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({file}): {e}")

    if not X_au_list:
        print("[ERROR] 유효한 sliding window가 생성되지 않았습니다.")
        return None, None, None, None, None

    print("[INFO] 리스트를 numpy 배열로 변환 중...")

    X_au_arr = np.array(X_au_list, dtype=np.float32)
    X_pose_arr = np.array(X_pose_list, dtype=np.float32)
    X_vehicle_arr = np.array(X_vehicle_list, dtype=np.float32)
    X_gaze_arr = np.array(X_gaze_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)

    del X_au_list, X_pose_list, X_vehicle_list, X_gaze_list, y_list
    gc.collect()

    return X_au_arr, X_pose_arr, X_vehicle_arr, X_gaze_arr, y_arr


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
        mean_feat,
        std_feat,
        min_feat,
        max_feat,
        median_feat,
        last_feat,
        delta_feat,
        diff_mean_feat,
        abs_diff_mean_feat
    ], axis=1).astype(np.float32)

    return X_static


def assemble_mode_features(feature_blocks: dict, mode: str, indices: np.ndarray) -> np.ndarray:
    blocks = mode.split('+')
    return np.hstack([feature_blocks[b][indices] for b in blocks]).astype(np.float32)


# =========================================================
# GLOBAL UNDERSAMPLING
# =========================================================
def balance_classes_global(X, y, num_classes=4, seed=42):
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
# BALANCED VALIDATION
# =========================================================
def balance_validation_set_by_class(X_val, y_val, seed=42, num_classes=4):
    rng = np.random.default_rng(seed)

    class_indices = [np.where(y_val == cls)[0] for cls in range(num_classes)]
    non_empty_sizes = [len(idx) for idx in class_indices if len(idx) > 0]

    if len(non_empty_sizes) <= 1:
        return X_val, y_val

    min_count = min(non_empty_sizes)

    selected_indices = []
    for cls in range(num_classes):
        cls_idx = class_indices[cls]
        if len(cls_idx) > 0:
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices, dtype=np.int32)
    rng.shuffle(selected_indices)

    return X_val[selected_indices], y_val[selected_indices]


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

        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
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


def evaluate_model_performance(
    model, X_val, y_val, class_names, fold_no,
    save_dir, mode_name, log_file
):
    num_classes = len(class_names)
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    bacc = balanced_accuracy_score(y_val, y_pred)
    prec = precision_score(
        y_val, y_pred,
        labels=np.arange(num_classes),
        average='macro',
        zero_division=0
    )
    rec = recall_score(
        y_val, y_pred,
        labels=np.arange(num_classes),
        average='macro',
        zero_division=0
    )
    f1 = f1_score(
        y_val, y_pred,
        labels=np.arange(num_classes),
        average='macro',
        zero_division=0
    )

    try:
        auc_score = roc_auc_score(
            y_val,
            y_proba,
            multi_class='ovr',
            average='macro'
        )
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

    log_msg = (
        f"   [{mode_name}] "
        f"Acc: {acc:.4f}, BAcc: {bacc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
        f"F1: {f1:.4f}, Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    )
    write_log(log_file, log_msg)

    safe_mode_name = sanitize_filename(mode_name)

    if Config.SAVE_FOLD_CM:
        cm_path = os.path.join(save_dir, f"CM_{safe_mode_name}_fold_{fold_no}.png")
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            save_path=cm_path,
            title=f'Fold {fold_no} - {mode_name} (balanced val) CM'
        )

    if Config.SAVE_FOLD_ROC:
        roc_path = os.path.join(save_dir, f"ROC_{safe_mode_name}_fold_{fold_no}.png")
        plot_multiclass_roc_curve(
            y_true=y_val,
            y_proba=y_proba,
            class_names=class_names,
            save_path=roc_path,
            title=f'Fold {fold_no} - {mode_name} (balanced val) ROC'
        )

    return {
        'acc': acc,
        'bacc': bacc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'spec': spec,
        'auc': auc_score
    }, y_val, y_pred, y_proba


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

    plt.bar(x - width/2, acc_means, width, yerr=acc_stds,
            label='Accuracy', capsize=5, color='skyblue', edgecolor='black')
    plt.bar(x + width/2, f1_means, width, yerr=f1_stds,
            label='Macro F1', capsize=5, color='steelblue', edgecolor='black')

    plt.xlabel('Modes (Feature Combinations)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'K-Fold Summary: {experiment_name}', fontsize=14)
    plt.xticks(x, modes, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(modes)):
        plt.text(x[i] - width/2, acc_means[i] + 0.015, f"{acc_means[i]:.2f}",
                 ha='center', fontsize=9)
        plt.text(x[i] + width/2, f1_means[i] + 0.015, f"{f1_means[i]:.2f}",
                 ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "KFold_Summary_XGBoost_GlobalUndersamplingBeforeKFold.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[INFO] 결과 그래프 저장 완료: {save_path}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_seeds()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    LOG_FILE = os.path.join(Config.RESULTS_DIR, "training_result_log_xgboost.txt")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("Experiment Start: XGBoost Multi-Class Classification\n")
        f.write("Split Method    : StratifiedKFold\n")
        f.write("Train Balancing : global undersampling BEFORE KFold\n")
        f.write("Val Evaluation  : balanced validation only\n")
        f.write(f"Classes: {Config.CLASS_NAMES}\n")
        f.write(f"Modes: {Config.MODES}\n")
        f.write("=" * 80 + "\n")

    print("=" * 80)
    print(f"🚀 STARTING XGBOOST EXPERIMENT: {Config.CLASS_NAMES}")
    print("=" * 80)

    X_au_seq, X_pose_seq, X_vehicle_seq, X_gaze_seq, y = load_and_create_sliding_window_data()

    if X_au_seq is None:
        print("학습을 진행할 데이터가 충분하지 않습니다.")
        raise SystemExit

    print(f"\n[INFO] Sliding Window 생성 완료")
    print(f" - AU shape      : {X_au_seq.shape}")
    print(f" - Pose shape    : {X_pose_seq.shape}")
    print(f" - Vehicle shape : {X_vehicle_seq.shape}")
    print(f" - Gaze shape    : {X_gaze_seq.shape}")
    print(f" - Labels shape  : {y.shape}")
    print(f" - Class Distribution (Before Global Balancing): {format_class_distribution(y, Config.CLASS_NAMES)}")

    write_log(LOG_FILE, f"Total samples before balancing: {len(y)}")
    write_log(LOG_FILE, f"Class Distribution before balancing: {format_class_distribution(y, Config.CLASS_NAMES)}")

    print("\n[INFO] XGBoost용 윈도우 요약 feature 생성 중...")
    X_au_static = summarize_sequence_features(X_au_seq)
    X_pose_static = summarize_sequence_features(X_pose_seq)
    X_vehicle_static = summarize_sequence_features(X_vehicle_seq)
    X_gaze_static = summarize_sequence_features(X_gaze_seq)

    feature_blocks_raw = {
        'AU': X_au_static,
        'Pose': X_pose_static,
        'Vehicle': X_vehicle_static,
        'Gaze': X_gaze_static
    }

    print("[INFO] Static Feature Shapes:")
    for k, v in feature_blocks_raw.items():
        print(f" - {k:<7}: {v.shape}")

    del X_au_seq, X_pose_seq, X_vehicle_seq, X_gaze_seq
    gc.collect()

    metric_names = ['acc', 'bacc', 'prec', 'rec', 'f1', 'spec', 'auc']
    final_results = {
        mode: {metric: [] for metric in metric_names}
        for mode in Config.MODES
    }

    all_predictions = {
        mode: {'y_true': [], 'y_pred': [], 'y_proba': []}
        for mode in Config.MODES
    }

    # -------------------------------------------------
    # 모드별 global undersampling 먼저 수행
    # -------------------------------------------------
    balanced_mode_data = {}

    for mode in Config.MODES:
        X_mode_full = assemble_mode_features(
            feature_blocks_raw,
            mode,
            np.arange(len(y))
        )

        X_mode_bal, y_bal = balance_classes_global(
            X_mode_full,
            y,
            num_classes=len(Config.CLASS_NAMES),
            seed=Config.SEED
        )

        balanced_mode_data[mode] = {
            'X': X_mode_bal,
            'y': y_bal
        }

        write_log(LOG_FILE, f"[{mode}] Global balanced class distribution: {format_class_distribution(y_bal, Config.CLASS_NAMES)}")
        write_log(LOG_FILE, f"[{mode}] Samples: {len(y)} -> {len(y_bal)}")

    # -------------------------------------------------
    # KFold
    # -------------------------------------------------
    y_balanced_reference = balanced_mode_data['AU']['y']

    skf = StratifiedKFold(
        n_splits=Config.N_SPLITS,
        shuffle=True,
        random_state=Config.SEED
    )

    fold_no = 1
    for train_idx, val_idx in skf.split(balanced_mode_data['AU']['X'], y_balanced_reference):
        print("\n" + "-" * 80)
        print(f"Fold {fold_no}/{Config.N_SPLITS}")
        print("-" * 80)

        train_dist = format_class_distribution(y_balanced_reference[train_idx], Config.CLASS_NAMES)
        val_dist = format_class_distribution(y_balanced_reference[val_idx], Config.CLASS_NAMES)

        write_log(LOG_FILE, f"\n--- Fold {fold_no}/{Config.N_SPLITS} ---")
        write_log(LOG_FILE, f"Train Distribution: {train_dist}")
        write_log(LOG_FILE, f"Val Distribution  : {val_dist}")

        for mode in Config.MODES:
            mode_save_dir = os.path.join(Config.RESULTS_DIR, sanitize_filename(mode))
            os.makedirs(mode_save_dir, exist_ok=True)

            print(f" > Training XGBoost Mode: {mode}")

            X_mode = balanced_mode_data[mode]['X']
            y_mode = balanced_mode_data[mode]['y']

            X_train = X_mode[train_idx]
            X_val = X_mode[val_idx]
            y_train = y_mode[train_idx]
            y_val = y_mode[val_idx]

            if Config.USE_SCALER:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_fit = X_train
                X_val_scaled = X_val

            X_val_bal, y_val_bal = balance_validation_set_by_class(
                X_val_scaled,
                y_val,
                seed=Config.SEED + 1000 + fold_no,
                num_classes=len(Config.CLASS_NAMES)
            )

            bal_val_dist = format_class_distribution(y_val_bal, Config.CLASS_NAMES)
            write_log(LOG_FILE, f"   [{mode}] Balanced Val Distribution: {bal_val_dist}")
            write_log(LOG_FILE, f"   [{mode}] Val Samples: {len(y_val)} -> {len(y_val_bal)}")

            model = XGBClassifier(**Config.XGB_PARAMS)
            model.fit(X_train_fit, y_train)

            metrics, y_true_fold, y_pred_fold, y_proba_fold = evaluate_model_performance(
                model=model,
                X_val=X_val_bal,
                y_val=y_val_bal,
                class_names=Config.CLASS_NAMES,
                fold_no=fold_no,
                save_dir=mode_save_dir,
                mode_name=mode,
                log_file=LOG_FILE
            )

            for metric in metric_names:
                final_results[mode][metric].append(metrics[metric])

            all_predictions[mode]['y_true'].extend(y_true_fold.tolist())
            all_predictions[mode]['y_pred'].extend(y_pred_fold.tolist())
            all_predictions[mode]['y_proba'].append(y_proba_fold)

            del X_train, X_val, X_train_fit, X_val_scaled, X_val_bal
            del y_train, y_val, y_val_bal, model
            gc.collect()

        fold_no += 1

    print("\n" + "=" * 80)
    print("📊 FINAL XGBOOST SUMMARY [GLOBAL UNDERSAMPLING BEFORE KFOLD]")
    print("=" * 80)

    write_log(LOG_FILE, "\n" + "=" * 80, print_console=False)
    write_log(LOG_FILE, "FINAL XGBOOST SUMMARY [GLOBAL UNDERSAMPLING BEFORE KFOLD]", print_console=False)
    write_log(LOG_FILE, "=" * 80, print_console=False)

    for mode in Config.MODES:
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
        mode_save_dir = os.path.join(Config.RESULTS_DIR, sanitize_filename(mode))
        y_true_all = np.array(all_predictions[mode]['y_true'], dtype=np.int32)
        y_pred_all = np.array(all_predictions[mode]['y_pred'], dtype=np.int32)
        y_proba_all = np.vstack(all_predictions[mode]['y_proba'])

        if Config.SAVE_COMBINED_CM:
            cm_all = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(Config.CLASS_NAMES)))
            cm_path = os.path.join(mode_save_dir, f"Combined_CM_{sanitize_filename(mode)}.png")
            plot_confusion_matrix(
                cm=cm_all,
                class_names=Config.CLASS_NAMES,
                save_path=cm_path,
                title=f'Total {Config.N_SPLITS}-Fold Combined CM - {mode} [global undersampling before KFold]'
            )

        if Config.SAVE_COMBINED_ROC:
            roc_path = os.path.join(mode_save_dir, f"Combined_ROC_{sanitize_filename(mode)}.png")
            plot_multiclass_roc_curve(
                y_true=y_true_all,
                y_proba=y_proba_all,
                class_names=Config.CLASS_NAMES,
                save_path=roc_path,
                title=f'Total {Config.N_SPLITS}-Fold Combined ROC - {mode} [global undersampling before KFold]'
            )

    plot_kfold_summary(
        final_results=final_results,
        save_dir=Config.RESULTS_DIR,
        experiment_name="XGBoost Multi-Class (Global Undersampling Before KFold)"
    )

    print(f"\n[INFO] 모든 결과가 저장되었습니다: {Config.RESULTS_DIR}")
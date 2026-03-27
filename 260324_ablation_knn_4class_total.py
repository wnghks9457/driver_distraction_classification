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
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# USER CONFIGURATION
# =========================================================
class Config:
    FOLDER_PATH = "Distraction_dataset_Final_Merged"
    RESULTS_DIR = "260324_Results_KNN_MultiExperiment_Ablation"

    SEED = 42

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

    KNN_PARAMS = {
        'n_neighbors': 9,
        'weights': 'uniform',
        'metric': 'minkowski',
        'p': 2,
        'n_jobs': -1
    }

    EXPERIMENTS = {
        '4Class_ND_CD_ED_MD': {
            'TARGET_LABELS_MAP': {0: 0, 1: 1, 2: 2, 3: 3},
            'CLASS_NAMES': ['ND', 'CD', 'ED', 'MD']
        },
        '3Class_ND_ED_MD': {
            'TARGET_LABELS_MAP': {0: 0, 2: 1, 3: 2},
            'CLASS_NAMES': ['ND', 'ED', 'MD']
        },
        '3Class_ND_CDED_MD': {
            'TARGET_LABELS_MAP': {0: 0, 1: 1, 2: 1, 3: 2},
            'CLASS_NAMES': ['ND', 'CDED', 'MD']
        },
        '3Class_ND_CD_MD': {
            'TARGET_LABELS_MAP': {0: 0, 1: 1, 3: 2},
            'CLASS_NAMES': ['ND', 'CD', 'MD']
        }
    }

    FEATURE_GROUPS = {
        'AU': [
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
            'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        ],
        'POSE': [
            'pose_Tx', 'pose_Ty', 'pose_Tz',
            'pose_Rx', 'pose_Ry', 'pose_Rz'
        ],
        'VEHICLE': ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset'],
        'GAZE': [
            'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
            'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
            'gaze_angle_x', 'gaze_angle_y',
            'gaze_vel', 'gaze_amp', 'gaze_acc'
        ]
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


def infer_raw_label_from_filename(filename: str) -> int:
    if '-005' in filename:
        return 1  # CD
    elif '-006' in filename:
        return 2  # ED
    elif '-007' in filename:
        return 3  # MD
    else:
        return 0  # ND


# =========================================================
# FEATURE / ABLATION HELPERS
# =========================================================
def get_feature_groups():
    au_features = Config.FEATURE_GROUPS['AU']
    pose_features = Config.FEATURE_GROUPS['POSE']
    vehicle_features = Config.FEATURE_GROUPS['VEHICLE']

    gaze_raw_features = [
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y'
    ]
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']
    gaze_features = Config.FEATURE_GROUPS['GAZE']

    return au_features, pose_features, vehicle_features, gaze_raw_features, kinematic_features, gaze_features


def get_all_features_from_groups(feature_groups):
    all_features = []
    for _, feats in feature_groups.items():
        all_features.extend(feats)
    return all_features


def build_ablation_configs():
    feature_groups = Config.FEATURE_GROUPS
    all_features = get_all_features_from_groups(feature_groups)

    ablation_configs = {}

    ablation_configs['baseline_all_features'] = {
        'selected_groups': list(feature_groups.keys()),
        'selected_features': all_features
    }

    for group_name in feature_groups.keys():
        selected_groups = [g for g in feature_groups.keys() if g != group_name]
        selected_features = []
        for g in selected_groups:
            selected_features.extend(feature_groups[g])

        ablation_configs[f'group_minus_{group_name}'] = {
            'selected_groups': selected_groups,
            'selected_features': selected_features
        }

    for feat in all_features:
        selected_features = [f for f in all_features if f != feat]

        selected_groups = []
        for g, feats in feature_groups.items():
            if any(f in selected_features for f in feats):
                selected_groups.append(g)

        ablation_configs[f'feature_minus_{feat}'] = {
            'selected_groups': selected_groups,
            'selected_features': selected_features
        }

    return ablation_configs


def get_group_name_of_feature(feature_name):
    for group_name, feats in Config.FEATURE_GROUPS.items():
        if feature_name in feats:
            return group_name
    return "UNKNOWN"


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_subject_data(df, au_features, gaze_features):
    df = df.copy()
    sampling_rate = Config.FPS

    if 'timestamp' in df.columns and len(df) > 1:
        dt = df['timestamp'].diff().replace(0, np.nan)
        mean_dt = dt.mean()
        if pd.isna(mean_dt) or mean_dt <= 0:
            mean_dt = 1.0 / sampling_rate
        dt = dt.fillna(mean_dt).astype(np.float32)
    else:
        dt = pd.Series(np.full(len(df), 1.0 / sampling_rate, dtype=np.float32), index=df.index)

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
        target_cols = [c for c in (au_features + gaze_features) if c in df.columns]
        if target_cols:
            baseline_mean = nd_data[target_cols].mean()
            df[target_cols] = df[target_cols] - baseline_mean

    return df


# =========================================================
# DYNAMIC DATA LOADING (selected_features 반영)
# =========================================================
def create_windows_from_file_list(file_list, target_labels_map, selected_features):
    target_time_steps = Config.TIME_STEPS
    au_features, pose_features, vehicle_features, gaze_raw_features, kinematic_features, gaze_features = get_feature_groups()

    final_features = list(selected_features)

    generated_features = {'gaze_vel', 'gaze_amp', 'gaze_acc'}
    columns_to_load = [f for f in final_features if f not in generated_features]

    for col in ['gaze_angle_x', 'gaze_angle_y']:
        if col not in columns_to_load:
            columns_to_load.append(col)

    if 'Distraction' not in columns_to_load:
        columns_to_load.append('Distraction')

    X_seq_list = []
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
                gaze_features=gaze_features
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

                if raw_label not in target_labels_map:
                    continue

                final_label = target_labels_map[raw_label]

                X_seq_list.append(
                    pad_or_truncate_sequence(window_df[final_features].values, target_time_steps)
                )
                y_list.append(final_label)

        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({file}): {e}")

    if not X_seq_list:
        return None, None

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32)
    )


# =========================================================
# FEATURE ENGINEERING (XGBoost와 동일 통계량)
# =========================================================
def summarize_sequence_features(X_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    X_seq: shape = (N, T, F)
    반환: shape = (N, F * 5)
          [mean, var, std, skewness, kurtosis]
    """
    mean_feat = np.mean(X_seq, axis=1)
    var_feat = np.var(X_seq, axis=1)
    std_feat = np.std(X_seq, axis=1)

    centered = X_seq - mean_feat[:, np.newaxis, :]
    skew_feat = np.mean(centered ** 3, axis=1) / (std_feat ** 3 + eps)

    kurt_feat = np.mean(centered ** 4, axis=1) / (std_feat ** 4 + eps)
    kurt_feat = kurt_feat - 3.0

    X_static = np.concatenate([
        mean_feat,
        var_feat,
        std_feat,
        skew_feat,
        kurt_feat
    ], axis=1).astype(np.float32)

    return X_static


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

    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        if y_proba.ndim == 1:
            pos_proba = y_proba
        elif y_proba.shape[1] == 1:
            pos_proba = y_proba[:, 0]
        else:
            pos_proba = y_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color='blue', lw=2,
            label=f'{class_names[1]} vs {class_names[0]} (AUC = {roc_auc:.2f})'
        )

    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

        valid_curve_count = 0
        for i, color in zip(range(n_classes), colors):
            if np.sum(y_true_bin[:, i]) == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
            )
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

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_val, y_pred)
    bacc = balanced_accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)
    rec = recall_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0)

    try:
        if num_classes == 2:
            auc_score = roc_auc_score(y_val, y_proba[:, 1])
        else:
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

    log_msg = (
        f"   [{mode_name}] Acc: {acc:.4f}, BAcc: {bacc:.4f}, "
        f"Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, "
        f"Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    )
    write_log(log_file, log_msg)

    safe_mode_name = sanitize_filename(mode_name)

    if Config.SAVE_FOLD_CM:
        cm_path = os.path.join(save_dir, f"CM_{safe_mode_name}_fold_{fold_no}.png")
        plot_confusion_matrix(cm, class_names, cm_path, f'Fold {fold_no} - {mode_name} CM')

    if Config.SAVE_FOLD_ROC:
        roc_path = os.path.join(save_dir, f"ROC_{safe_mode_name}_fold_{fold_no}.png")
        plot_multiclass_roc_curve(y_val, y_proba, class_names, roc_path, f'Fold {fold_no} - {mode_name} ROC')

    return {
        'acc': acc,
        'bacc': bacc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'spec': spec,
        'auc': auc_score
    }, y_val, y_pred, y_proba


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_seeds()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    print("=" * 100)
    print("🚀 STARTING KNN MULTI-EXPERIMENT ABLATION (FILE-BASED STRATIFIED SPLIT)")
    print("=" * 100)

    folder_path = Config.FOLDER_PATH
    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        raise SystemExit

    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    if len(all_files) == 0:
        print(f"[ERROR] CSV 파일이 없습니다: {folder_path}")
        raise SystemExit

    ablation_configs = build_ablation_configs()
    summary_results = []

    for exp_name, exp_cfg in Config.EXPERIMENTS.items():
        print("\n" + "#" * 100)
        print(f"🧪 EXPERIMENT: {exp_name}")
        print("#" * 100)

        target_labels_map = exp_cfg['TARGET_LABELS_MAP']
        class_names = exp_cfg['CLASS_NAMES']
        num_classes = len(class_names)

        selected_file_paths = []
        selected_file_labels = []

        for f in all_files:
            raw_file_label = infer_raw_label_from_filename(f)

            if 0 in target_labels_map:
                selected_file_paths.append(os.path.join(folder_path, f))

                if raw_file_label in target_labels_map:
                    selected_file_labels.append(target_labels_map[raw_file_label])
                else:
                    selected_file_labels.append(target_labels_map[0])

            else:
                if raw_file_label in target_labels_map:
                    selected_file_paths.append(os.path.join(folder_path, f))
                    selected_file_labels.append(target_labels_map[raw_file_label])

        file_paths = np.array(selected_file_paths)
        file_labels = np.array(selected_file_labels, dtype=np.int32)

        if len(file_paths) == 0:
            write_log(os.path.join(Config.RESULTS_DIR, "tmp.log"), f"[WARNING] {exp_name}: 실험 대상 파일이 없습니다.", print_console=True)
            continue

        unique_labels, counts = np.unique(file_labels, return_counts=True)
        min_class_count = counts.min()

        if len(unique_labels) < 2 or min_class_count < 2:
            print(f"[WARNING] {exp_name}: StratifiedKFold를 수행하기 위한 파일 수가 부족합니다.")
            continue

        n_splits = min(Config.N_SPLITS, int(min_class_count))
        skf_files = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=Config.SEED
        )

        for abl_name, abl_cfg in ablation_configs.items():
            selected_features = abl_cfg['selected_features']
            selected_groups = abl_cfg['selected_groups']

            exp_result_dir = os.path.join(
                Config.RESULTS_DIR,
                sanitize_filename(exp_name),
                sanitize_filename(abl_name)
            )
            os.makedirs(exp_result_dir, exist_ok=True)

            log_file = os.path.join(
                exp_result_dir,
                f"training_result_log_{sanitize_filename(exp_name)}_{sanitize_filename(abl_name)}.txt"
            )

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Experiment Start: {exp_name}\n")
                f.write(f"Ablation        : {abl_name}\n")
                f.write("Model           : KNN\n")
                f.write("Split Method    : File-Based StratifiedKFold (Leakage Free)\n")
                f.write("Train Balancing : In-Fold Undersampling\n")
                f.write(f"Classes         : {class_names}\n")
                f.write(f"Target Map      : {target_labels_map}\n")
                f.write(f"Selected Groups : {selected_groups}\n")
                f.write(f"Selected Features ({len(selected_features)}): {selected_features}\n")
                f.write(f"KNN Params      : {Config.KNN_PARAMS}\n")
                f.write("=" * 80 + "\n")

            print("\n" + "=" * 100)
            print(f"🚀 STARTING KNN EXPERIMENT: {exp_name}")
            print(f"🧪 Ablation: {abl_name}")
            print(f"🧩 Selected Groups: {selected_groups}")
            print(f"🧩 Num Features: {len(selected_features)}")
            print("=" * 100)

            write_log(log_file, f"Filtered File Distribution: {format_class_distribution(file_labels, class_names)}", print_console=False)

            metric_names = ['acc', 'bacc', 'prec', 'rec', 'f1', 'spec', 'auc']
            final_results = {metric: [] for metric in metric_names}
            all_predictions = {'y_true': [], 'y_pred': [], 'y_proba': []}

            for fold_no, (train_f_idx, val_f_idx) in enumerate(skf_files.split(file_paths, file_labels), 1):
                print("\n" + "-" * 80)
                print(f"{exp_name} | {abl_name} | Fold {fold_no}/{n_splits} Data Loading & Processing")
                print("-" * 80)

                train_files = file_paths[train_f_idx]
                val_files = file_paths[val_f_idx]

                write_log(log_file, f"\n--- Fold {fold_no}/{n_splits} ---")
                write_log(log_file, f"Train Files Count: {len(train_files)} | Val Files Count: {len(val_files)}")

                print("[INFO] 학습용 파일 윈도우 생성 중...")
                X_tr_seq, y_tr_raw = create_windows_from_file_list(
                    train_files, target_labels_map, selected_features
                )

                print("[INFO] 검증용 파일 윈도우 생성 중...")
                X_val_seq, y_val_raw = create_windows_from_file_list(
                    val_files, target_labels_map, selected_features
                )

                if X_tr_seq is None or X_val_seq is None:
                    write_log(log_file, "[WARNING] 이번 Fold에서 생성된 윈도우가 없습니다. 건너뜁니다.")
                    continue

                if len(np.unique(y_tr_raw)) < num_classes or len(np.unique(y_val_raw)) < num_classes:
                    write_log(log_file, "[WARNING] Train/Val 윈도우가 모든 클래스를 포함하지 않아 Fold를 건너뜁니다.")
                    continue

                write_log(log_file, f"Train Windows: {len(y_tr_raw)} | Val Windows: {len(y_val_raw)}")
                write_log(log_file, f"Train Distribution (raw): {format_class_distribution(y_tr_raw, class_names)}")
                write_log(log_file, f"Val Distribution (raw): {format_class_distribution(y_val_raw, class_names)}")

                X_tr_mode = summarize_sequence_features(X_tr_seq)
                X_val_mode = summarize_sequence_features(X_val_seq)

                del X_tr_seq, X_val_seq
                gc.collect()

                X_tr_bal, y_tr_bal = balance_classes_by_count(
                    X_tr_mode, y_tr_raw, seed=Config.SEED, num_classes=num_classes
                )
                X_val_bal, y_val_bal = balance_classes_by_count(
                    X_val_mode, y_val_raw, seed=Config.SEED + 1000 + fold_no, num_classes=num_classes
                )

                if len(y_tr_bal) == 0 or len(y_val_bal) == 0:
                    write_log(log_file, f"[WARNING] {abl_name} 언더샘플링 후 데이터가 없습니다.")
                    continue

                if len(np.unique(y_tr_bal)) < num_classes or len(np.unique(y_val_bal)) < num_classes:
                    write_log(log_file, f"[WARNING] {abl_name} 언더샘플링 후 일부 클래스가 없어 건너뜁니다.")
                    continue

                write_log(log_file, f"Train Distribution (balanced): {format_class_distribution(y_tr_bal, class_names)}")
                write_log(log_file, f"Val Distribution (balanced): {format_class_distribution(y_val_bal, class_names)}")

                if Config.USE_SCALER:
                    scaler = StandardScaler()
                    X_tr_final = scaler.fit_transform(X_tr_bal)
                    X_val_final = scaler.transform(X_val_bal)
                else:
                    X_tr_final = X_tr_bal
                    X_val_final = X_val_bal

                model = KNeighborsClassifier(**Config.KNN_PARAMS)
                model.fit(X_tr_final, y_tr_bal)

                metrics, y_true_fold, y_pred_fold, y_proba_fold = evaluate_model_performance(
                    model=model,
                    X_val=X_val_final,
                    y_val=y_val_bal,
                    class_names=class_names,
                    fold_no=fold_no,
                    save_dir=exp_result_dir,
                    mode_name=abl_name,
                    log_file=log_file
                )

                for metric in metric_names:
                    final_results[metric].append(metrics[metric])

                all_predictions['y_true'].extend(y_true_fold.tolist())
                all_predictions['y_pred'].extend(y_pred_fold.tolist())
                all_predictions['y_proba'].append(y_proba_fold)

                del model, X_tr_mode, X_val_mode, X_tr_bal, X_val_bal, X_tr_final, X_val_final
                gc.collect()

            print("\n" + "=" * 80)
            print(f"📊 FINAL KNN SUMMARY [{exp_name} | {abl_name}]")
            print("=" * 80)
            write_log(log_file, "\n" + "=" * 80, print_console=False)
            write_log(log_file, f"FINAL KNN SUMMARY [{exp_name} | {abl_name}]", print_console=False)
            write_log(log_file, "=" * 80, print_console=False)

            if not final_results['acc']:
                print("[WARNING] 유효한 fold 결과가 없습니다.")
                continue

            summary_lines = [
                f"[{abl_name}]",
                f"  ACC : {np.mean(final_results['acc']):.4f} (+/- {np.std(final_results['acc']):.4f})",
                f"  BACC: {np.mean(final_results['bacc']):.4f} (+/- {np.std(final_results['bacc']):.4f})",
                f"  PREC: {np.mean(final_results['prec']):.4f} (+/- {np.std(final_results['prec']):.4f})",
                f"  REC : {np.mean(final_results['rec']):.4f} (+/- {np.std(final_results['rec']):.4f})",
                f"  F1  : {np.mean(final_results['f1']):.4f} (+/- {np.std(final_results['f1']):.4f})",
                f"  SPEC: {np.mean(final_results['spec']):.4f} (+/- {np.std(final_results['spec']):.4f})",
                f"  AUC : {np.mean(final_results['auc']):.4f} (+/- {np.std(final_results['auc']):.4f})"
            ]
            summary_text = "\n".join(summary_lines)
            print(summary_text)
            write_log(log_file, summary_text, print_console=False)
            print("-" * 80)

            if all_predictions['y_true']:
                y_true_all = np.array(all_predictions['y_true'], dtype=np.int32)
                y_pred_all = np.array(all_predictions['y_pred'], dtype=np.int32)
                y_proba_all = np.vstack(all_predictions['y_proba'])

                if Config.SAVE_COMBINED_CM:
                    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(num_classes))
                    cm_path = os.path.join(exp_result_dir, f"Combined_CM_{sanitize_filename(abl_name)}.png")
                    plot_confusion_matrix(cm_all, class_names, cm_path, f'Total Combined CM - {abl_name}')

                if Config.SAVE_COMBINED_ROC:
                    roc_path = os.path.join(exp_result_dir, f"Combined_ROC_{sanitize_filename(abl_name)}.png")
                    plot_multiclass_roc_curve(
                        y_true_all, y_proba_all, class_names, roc_path,
                        f'Total Combined ROC - {abl_name}'
                    )

            removed_type = "baseline"
            removed_name = "none"
            removed_group = "none"

            if abl_name.startswith("group_minus_"):
                removed_type = "group"
                removed_name = abl_name.replace("group_minus_", "")
                removed_group = removed_name
            elif abl_name.startswith("feature_minus_"):
                removed_type = "feature"
                removed_name = abl_name.replace("feature_minus_", "")
                removed_group = get_group_name_of_feature(removed_name)

            summary_row = {
                'experiment': exp_name,
                'ablation': abl_name,
                'removed_type': removed_type,
                'removed_name': removed_name,
                'removed_group': removed_group,
                'num_features': len(selected_features),
                'acc_mean': np.mean(final_results['acc']),
                'acc_std': np.std(final_results['acc']),
                'bacc_mean': np.mean(final_results['bacc']),
                'bacc_std': np.std(final_results['bacc']),
                'prec_mean': np.mean(final_results['prec']),
                'prec_std': np.std(final_results['prec']),
                'rec_mean': np.mean(final_results['rec']),
                'rec_std': np.std(final_results['rec']),
                'f1_mean': np.mean(final_results['f1']),
                'f1_std': np.std(final_results['f1']),
                'spec_mean': np.mean(final_results['spec']),
                'spec_std': np.std(final_results['spec']),
                'auc_mean': np.mean(final_results['auc']),
                'auc_std': np.std(final_results['auc']),
            }
            summary_results.append(summary_row)

            pd.DataFrame(summary_results).to_csv(
                os.path.join(Config.RESULTS_DIR, "ablation_summary_knn.csv"),
                index=False,
                encoding='utf-8-sig'
            )

            print(f"[INFO] 실험 결과 저장 완료: {exp_result_dir}")

    print("\n" + "=" * 100)
    print(f"[INFO] 모든 실험이 완료되었습니다: {Config.RESULTS_DIR}")
    print("=" * 100)
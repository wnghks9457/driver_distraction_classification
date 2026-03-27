from operator import sub
import re
import gc
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold
from itertools import cycle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer, Concatenate, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence


# =========================================================
# [USER CONFIGURATION] 하이퍼 파라미터 및 설정
# =========================================================
class Config:
    # 경로 및 시드 설정 
    FOLDER_PATH = "Distraction_dataset_Final_Merged"
    RESULTS_DIR = "260324_Results_cnn-lstm_MultiExperiments_Ablation"
    SEED = 42

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

    # 데이터 전처리 (Sliding Window)
    FPS = 28
    WINDOW_SECONDS = 10
    STRIDE_SECONDS = 5
    TIME_STEPS = FPS * WINDOW_SECONDS
    STEP_SIZE = FPS * STRIDE_SECONDS

    # 모델 구조 파라미터
    CNN_FILTERS = 32
    CNN_KERNEL_SIZE = 3
    LSTM_UNITS = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.4

    # 학습 파라미터
    N_SPLITS = 5
    EPOCHS = 50
    BATCH_SIZE = 64

    # ===== Feature Group 정의 =====
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
# GPU 설정
# =========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] 사용 가능한 GPU: {len(gpus)}개")
    except RuntimeError as e:
        print(f"[ERROR] GPU 설정 오류: {e}")
else:
    print("[INFO] GPU를 찾을 수 없습니다. CPU로 실행합니다.")


# =========================================================
# 랜덤 시드 설정 함수
# =========================================================
def set_seeds(seed=Config.SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] 랜덤 시드 설정: {seed}")


# =========================================================
# 로그 기록용 헬퍼 함수
# =========================================================
def write_log(filepath, message, print_console=True):
    if print_console:
        print(message)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(message + "\n")


# =========================================================
# Feature / Ablation 관련 헬퍼
# =========================================================
def get_all_features_from_groups(feature_groups):
    all_features = []
    for _, feats in feature_groups.items():
        all_features.extend(feats)
    return all_features


def build_ablation_configs():
    feature_groups = Config.FEATURE_GROUPS
    all_features = get_all_features_from_groups(feature_groups)

    ablation_configs = {}

    # 0) Baseline
    ablation_configs['baseline_all_features'] = {
        'selected_groups': list(feature_groups.keys()),
        'selected_features': all_features
    }

    # 1) Group-wise ablation
    for group_name in feature_groups.keys():
        selected_groups = [g for g in feature_groups.keys() if g != group_name]
        selected_features = []
        for g in selected_groups:
            selected_features.extend(feature_groups[g])

        ablation_configs[f'group_minus_{group_name}'] = {
            'selected_groups': selected_groups,
            'selected_features': selected_features
        }

    # 2) Feature-wise ablation
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
# 1. 시선 운동학적 특성 계산 + AU/Gaze baseline correction
# =========================================================
def preprocess_subject_data(df, au_features, gaze_features):
    df = df.copy()
    sampling_rate = Config.FPS

    if 'timestamp' in df.columns:
        dt = df['timestamp'].diff()
        dt = dt.replace(0, np.nan)
        mean_dt = dt.mean() if not np.isnan(dt.mean()) else (1 / sampling_rate)
        dt = dt.fillna(mean_dt)
    else:
        dt = pd.Series(np.full(len(df), 1 / sampling_rate), index=df.index)

    # gaze kinematics 생성
    if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
        dx = df['gaze_angle_x'].diff()
        dy = df['gaze_angle_y'].diff()
        amp_rad = np.sqrt(dx**2 + dy**2).fillna(0)

        df['gaze_amp'] = np.degrees(amp_rad)
        df['gaze_vel'] = (df['gaze_amp'] / dt).fillna(0)

        d_vel = df['gaze_vel'].diff()
        df['gaze_acc'] = (d_vel / dt).fillna(0)

        if len(df) > 1:
            df.loc[df.index[:2], 'gaze_acc'] = 0
        elif len(df) == 1:
            df.loc[df.index[0], 'gaze_acc'] = 0
    else:
        df['gaze_amp'] = 0.0
        df['gaze_vel'] = 0.0
        df['gaze_acc'] = 0.0

    # AU + Gaze baseline correction
    nd_data = df[df['Distraction'] == 0]
    if not nd_data.empty:
        targets = [f for f in (au_features + gaze_features) if f in df.columns]
        means = nd_data[targets].mean()
        df[targets] = df[targets] - means

    return df


# =========================================================
# 2. 데이터 로드 (선택된 feature만 반영)
# =========================================================
def load_and_create_sliding_window_data(target_labels_map, class_names, selected_features):
    folder_path = Config.FOLDER_PATH
    target_time_steps = Config.TIME_STEPS

    print(f"[INFO] 타겟 클래스 매핑: {target_labels_map}")
    print(f"[INFO] 클래스 이름: {class_names}")

    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        return None, None, None, None, None

    csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])

    if not csv_files:
        print(f"[ERROR] CSV 파일이 없습니다.")
        return None, None, None, None, None

    au_features = Config.FEATURE_GROUPS['AU']
    pose_features = Config.FEATURE_GROUPS['POSE']
    vehicle_features = Config.FEATURE_GROUPS['VEHICLE']

    gaze_raw = [
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y'
    ]
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']
    gaze_features = Config.FEATURE_GROUPS['GAZE']

    final_features = [f for f in selected_features]

    # gaze kinematics는 preprocess에서 생성되므로 csv에서 직접 읽지 않음
    generated_features = {'gaze_vel', 'gaze_amp', 'gaze_acc'}
    columns_to_load = [f for f in final_features if f not in generated_features]

    # baseline correction / kinematics 생성에 필요한 gaze angle은 남겨둔다
    required_for_preprocess = ['gaze_angle_x', 'gaze_angle_y']
    for col in required_for_preprocess:
        if col not in columns_to_load:
            columns_to_load.append(col)

    # 항상 distraction 필요
    if 'Distraction' not in columns_to_load:
        columns_to_load.append('Distraction')

    # Feature Slicing (남아 있는 group만)
    feature_slices = []
    current_idx = 0
    used_group_info = []

    for group_name in ['AU', 'POSE', 'VEHICLE', 'GAZE']:
        group_feats = [f for f in Config.FEATURE_GROUPS[group_name] if f in final_features]
        if len(group_feats) > 0:
            feature_slices.append((current_idx, current_idx + len(group_feats)))
            used_group_info.append((group_name, group_feats))
            current_idx += len(group_feats)

    print(f"[INFO] Selected Features ({len(final_features)}): {final_features}")
    print(f"[INFO] Used Groups: {[g for g, _ in used_group_info]}")
    print(f"[INFO] Feature Slices: {feature_slices}")

    X_list, y_list, groups_list, subject_list = [], [], [], []

    for subject_id, file in enumerate(csv_files):
        try:
            try:
                df = pd.read_csv(file, usecols=columns_to_load + ['timestamp'])
            except ValueError:
                df = pd.read_csv(file, usecols=columns_to_load)

            df.fillna(0, inplace=True)

            # 파일별 실제 FPS 계산
            if 'timestamp' in df.columns and len(df) > 1:
                duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                actual_fps = len(df) / duration if duration > 0 else Config.FPS
            else:
                actual_fps = Config.FPS

            file_window_size = int(round(actual_fps * Config.WINDOW_SECONDS))
            file_step_size = int(round(actual_fps * Config.STRIDE_SECONDS))

            # preprocessing
            df = preprocess_subject_data(df, au_features, gaze_features)

            # 필요한 최종 feature 없으면 0으로 채움
            for col in final_features:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = df[col].astype('float32')

            values = df[final_features].values
            labels = df['Distraction'].values

            for i in range(0, len(df) - file_window_size + 1, file_step_size):
                raw_label = labels[i + file_window_size - 1]

                if raw_label in target_labels_map:
                    final_label = target_labels_map[raw_label]
                    X_window = values[i: i + file_window_size]

                    if len(X_window) < target_time_steps:
                        pad_width = target_time_steps - len(X_window)
                        X_window = np.pad(X_window, ((0, pad_width), (0, 0)), mode='edge')
                    elif len(X_window) > target_time_steps:
                        X_window = X_window[:target_time_steps]

                    X_list.append(X_window)
                    y_list.append(final_label)
                    groups_list.append(subject_id)

                    filename = os.path.basename(file)
                    match = re.search(r'T(\d+)-(\d+)', filename)
                    if match:
                        person_id = int(match.group(1))
                        subject_list.append(person_id)
                    else:
                        print(f"[WARNING] 파일명 형식을 파싱하지 못했습니다: {filename}")
                        subject_list.append(subject_id)

        except Exception as e:
            print(f"[WARNING] 파일 읽기 오류 ({file}): {e}")

    if not X_list:
        return None, None, None, None, None

    print("[INFO] 리스트를 Numpy 배열로 변환 중...")
    X_arr = np.array(X_list, dtype='float32')
    y_arr = np.array(y_list, dtype='int32')
    groups_arr = np.array(groups_list, dtype='int32')
    subject_arr = np.array(subject_list, dtype='int32')

    del X_list, y_list, groups_list, subject_list
    gc.collect()

    return X_arr, y_arr, groups_arr, subject_arr, feature_slices


# =========================================================
# Data Generator
# =========================================================
class DataGenerator(Sequence):
    def __init__(self, X_data, y_data, indices, batch_size, scaler=None, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.scaler = scaler
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        X_batch = self.X_data[batch_indices]
        y_batch = self.y_data[batch_indices]

        if self.scaler:
            b, t, f = X_batch.shape
            X_batch = self.scaler.transform(X_batch.reshape(-1, f)).reshape(b, t, f)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# 3. 모델 구성
# =========================================================
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced_channel = max(channel // self.ratio, 1)

        self.shared_layer_one = Dense(
            reduced_channel,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.shared_layer_two = Dense(
            channel,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        avg_pool = K.mean(x, axis=1)
        out = self.shared_layer_one(avg_pool)
        out = self.shared_layer_two(out)
        scale = K.expand_dims(out, axis=1)
        return x * scale

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config


class SoftAttention(Layer):
    def __init__(self, **kwargs):
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name='att_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='context_vector',
            shape=(feature_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        u_t = K.tanh(K.dot(x, self.W) + self.b)
        score = K.dot(u_t, self.u)
        a_t = K.softmax(score, axis=1)
        context = K.sum(x * a_t, axis=1)
        return context


def build_model(input_shape, feature_slices, num_classes):
    inputs = Input(shape=input_shape)
    encoded_branches = []

    for i, (start, end) in enumerate(feature_slices):
        x_slice = Lambda(lambda x, s=start, e=end: x[:, :, s:e], name=f'modality_slice_{i}')(inputs)

        x = Conv1D(
            filters=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            strides=1,
            padding='same',
            activation='relu',
            name=f'conv1_{i}'
        )(x_slice)

        x = Conv1D(
            filters=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            strides=1,
            padding='same',
            activation='relu',
            name=f'conv2_{i}'
        )(x)

        x = ChannelAttention(ratio=8, name=f'channel_att_{i}')(x)
        encoded_branches.append(x)

    if len(encoded_branches) > 1:
        x = Concatenate(axis=-1, name='modality_concat')(encoded_branches)
    else:
        x = encoded_branches[0]

    x = LSTM(units=Config.LSTM_UNITS, return_sequences=True, name='lstm_temporal')(x)
    x = Dropout(Config.DROPOUT_RATE, name='lstm_dropout')(x)
    x = SoftAttention(name='soft_attention')(x)
    x = Dense(Config.DENSE_UNITS, activation='relu', name='dense_fc')(x)
    x = Dropout(Config.DROPOUT_RATE, name='dense_dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# =========================================================
# 학습 곡선 시각화
# =========================================================
def plot_learning_curve(history, fold_no, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2)
    axes[0].set_title(f'Fold {fold_no} - Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(history.history['loss'], label='Train Loss', color='blue', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='orange', lw=2)
    axes[1].set_title(f'Fold {fold_no} - Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Learning_Curve_Fold_{fold_no}.png"), dpi=300)
    plt.close()


# =========================================================
# 평가 함수
# =========================================================
def evaluate_fold(model, val_gen, class_names, fold_no, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(val_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)

    y_true = []
    for i in range(len(val_gen)):
        _, batch_y = val_gen[i]
        y_true.extend(batch_y)
    y_true = np.array(y_true)

    y_pred = y_pred[:len(y_true)]
    y_pred_proba = y_pred_proba[:len(y_true)]

    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'{class_names[1]} vs {class_names[0]} (AUC = {roc_auc:.2f})')
    else:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        colors = cycle(['blue', 'red', 'green', 'orange'])

        for i, color in zip(range(n_classes), colors):
            if np.sum(y_true_bin[:, i]) == 0:
                print(f"[WARNING] Class {class_names[i]} has no positive samples in Fold {fold_no}.")
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold_no} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"ROC_Fold_{fold_no}.png"), dpi=300)
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        if n_classes == 2:
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = 0.0

    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = TN / (TN + FP)
        class_specificity = np.nan_to_num(class_specificity)
    spec = np.mean(class_specificity)

    log_msg = (
        f" [Result Fold {fold_no}] "
        f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
        f"F1: {f1:.4f}, Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    )
    write_log(log_file, log_msg)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Fold {fold_no} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"CM_Fold_{fold_no}.png"))
    plt.close()

    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'spec': spec,
        'auc': auc_score
    }, y_true, y_pred, y_pred_proba


# =========================================================
# Balancing 함수
# =========================================================
def balancing_classes(X, y, subjects: np.array, num_classes):
    X_balanced_list, y_balanced_list, subjects_balanced_list = [], [], []

    for sub in sorted(np.unique(subjects)):
        idx = np.where(subjects == sub)[0]
        X_sub = X[idx]
        y_sub = y[idx]
        sub_sub = subjects[idx]

        class_indices = [np.where(y_sub == cls)[0] for cls in range(num_classes)]
        non_empty_indices = [cls_idx for cls_idx in class_indices if len(cls_idx) > 0]

        if len(non_empty_indices) == 0:
            continue

        min_size = min(len(cls_idx) for cls_idx in non_empty_indices)
        selected_indices_list = []

        for cls_idx in class_indices:
            if len(cls_idx) > 0:
                selected = np.random.choice(cls_idx, min_size, replace=False)
                selected_indices_list.append(selected)

        balanced_indices = np.concatenate(selected_indices_list)
        balanced_indices.sort()

        X_balanced = X_sub[balanced_indices]
        y_balanced = y_sub[balanced_indices]
        subjects_balanced = sub_sub[balanced_indices]

        y_sub_sum = [sum(y_sub == i) for i in range(num_classes)]
        y_balanced_sum = [sum(y_balanced == i) for i in range(num_classes)]
        print(f"Subject {sub} - {y_sub_sum} → {y_balanced_sum}")

        X_balanced_list.append(X_balanced)
        y_balanced_list.append(y_balanced)
        subjects_balanced_list.append(subjects_balanced)

    X_new = np.concatenate(X_balanced_list, axis=0)
    y_new = np.concatenate(y_balanced_list, axis=0)
    subjects_new = np.concatenate(subjects_balanced_list, axis=0)

    return X_new, y_new, subjects_new


# =========================================================
# 메인
# =========================================================
if __name__ == "__main__":
    set_seeds()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    ablation_configs = build_ablation_configs()
    summary_results = []

    for exp_name, exp_cfg in Config.EXPERIMENTS.items():
        target_labels_map = exp_cfg['TARGET_LABELS_MAP']
        class_names = exp_cfg['CLASS_NAMES']
        num_classes = len(class_names)

        for abl_name, abl_cfg in ablation_configs.items():
            selected_features = abl_cfg['selected_features']
            selected_groups = abl_cfg['selected_groups']

            exp_result_dir = os.path.join(Config.RESULTS_DIR, exp_name, abl_name)
            exp_checkpoint_dir = os.path.join(exp_result_dir, "checkpoints")
            os.makedirs(exp_result_dir, exist_ok=True)
            os.makedirs(exp_checkpoint_dir, exist_ok=True)

            LOG_FILE = os.path.join(exp_result_dir, "training_result_log.txt")
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Experiment Start: {exp_name}\n")
                f.write(f"Classes: {class_names}\n")
                f.write(f"Ablation: {abl_name}\n")
                f.write(f"Selected Groups: {selected_groups}\n")
                f.write(f"Selected Features ({len(selected_features)}): {selected_features}\n")
                f.write("=" * 50 + "\n")

            print("=" * 100)
            print(f"🚀 Classification (Subject-Independent): {exp_name} | {class_names}")
            print(f"🧪 Ablation: {abl_name}")
            print(f"🧩 Selected Groups: {selected_groups}")
            print(f"🧩 Num Features: {len(selected_features)}")
            print("=" * 100)

            # 데이터 로드
            X, y, groups, subjects, feature_slices = load_and_create_sliding_window_data(
                target_labels_map=target_labels_map,
                class_names=class_names,
                selected_features=selected_features
            )

            if X is None:
                print("학습을 진행할 데이터가 충분하지 않습니다.")
                continue

            print(f"\n[INFO] Applying Global Balancing...")
            print(f"Before Balancing: Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            X, y, subjects = balancing_classes(X, y, subjects, num_classes)
            print(f"After Balancing: Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"Total Samples: {X.shape[0]}, Subjects: {len(np.unique(subjects))}")
            print("=" * 50)
            print(f"\n[Data Info] Input Shape: {X.shape}")

            sgkf = StratifiedGroupKFold(
                n_splits=Config.N_SPLITS,
                shuffle=True,
                random_state=Config.SEED
            )

            all_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'spec': [], 'auc': []}
            y_true_all = []
            y_pred_all = []
            y_pred_proba_all = []

            fold_no = 1

            for train_idx, val_idx in sgkf.split(X, y, groups=subjects):
                print(f"\nTraining Fold {fold_no}...")

                # Train balancing
                y_train_fold = y[train_idx]
                unique_classes, class_counts = np.unique(y_train_fold, return_counts=True)
                min_count = class_counts.min()
                print(f" > Train Class Balance Adjustment: Target count per class = {min_count}")

                balanced_train_indices = []
                for cls in unique_classes:
                    cls_indices = train_idx[y[train_idx] == cls]
                    selected_indices = np.random.choice(cls_indices, min_count, replace=False)
                    balanced_train_indices.extend(selected_indices)

                balanced_train_idx = np.array(balanced_train_indices)
                np.random.shuffle(balanced_train_idx)

                # Validation balancing
                y_val_fold = y[val_idx]
                val_unique_classes, val_class_counts = np.unique(y_val_fold, return_counts=True)
                val_min_count = val_class_counts.min()
                print(f" > Val Class Balance Adjustment: Target count per class = {val_min_count}")

                balanced_val_indices = []
                for cls in val_unique_classes:
                    cls_indices = val_idx[y[val_idx] == cls]
                    selected_indices = np.random.choice(cls_indices, val_min_count, replace=False)
                    balanced_val_indices.extend(selected_indices)

                balanced_val_idx = np.array(balanced_val_indices)
                np.random.shuffle(balanced_val_idx)

                # Scaler fit
                scaler = MinMaxScaler()
                try:
                    X_train_flat = X[balanced_train_idx].reshape(-1, X.shape[-1])
                    scaler.fit(X_train_flat)
                    del X_train_flat
                    gc.collect()
                except MemoryError:
                    print("[WARNING] Scaler fit 중 메모리 부족. 샘플링하여 fit 진행.")
                    sample_size = max(len(balanced_train_idx) // 10, 1)
                    sample_idx = np.random.choice(balanced_train_idx, size=sample_size, replace=False)
                    X_sample_flat = X[sample_idx].reshape(-1, X.shape[-1])
                    scaler.fit(X_sample_flat)
                    del X_sample_flat
                    gc.collect()

                train_gen = DataGenerator(X, y, balanced_train_idx, Config.BATCH_SIZE, scaler=scaler, shuffle=True)
                val_gen = DataGenerator(X, y, balanced_val_idx, Config.BATCH_SIZE, scaler=scaler, shuffle=False)

                if len(train_gen) == 0 or len(val_gen) == 0:
                    print("[WARNING] 배치 수가 0입니다. BATCH_SIZE를 줄이거나 데이터 수를 확인하세요.")
                    break

                model = build_model((X.shape[1], X.shape[2]), feature_slices, num_classes)

                checkpoint_path = os.path.join(
                    exp_checkpoint_dir,
                    f"best_model_fold_{fold_no}.weights.h5"
                )

                model_checkpoint = ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )

                history = model.fit(
                    train_gen,
                    epochs=Config.EPOCHS,
                    validation_data=val_gen,
                    callbacks=[model_checkpoint],
                    verbose=1
                )

                model.load_weights(checkpoint_path)
                print(f"[INFO] Best weights loaded from: {checkpoint_path}")

                plot_learning_curve(history, fold_no, exp_result_dir)

                fold_res, y_t, y_p, y_proba = evaluate_fold(
                    model, val_gen, class_names, fold_no, exp_result_dir, LOG_FILE
                )

                for k in all_metrics:
                    all_metrics[k].append(fold_res[k])

                y_true_all.extend(y_t)
                y_pred_all.extend(y_p)
                y_pred_proba_all.extend(y_proba)

                del model, train_gen, val_gen, scaler
                K.clear_session()
                gc.collect()

                fold_no += 1

            if len(all_metrics['acc']) == 0:
                print("[WARNING] 유효한 fold 결과가 없어 summary 저장을 건너뜁니다.")
                continue

            print("\n" + "=" * 50)
            print(f"Final {len(all_metrics['acc'])}-Fold Average Performance ({class_names})")
            print("=" * 50)

            write_log(LOG_FILE, "\n" + "=" * 50, print_console=False)
            write_log(LOG_FILE, f"Final {len(all_metrics['acc'])}-Fold Average Performance ({class_names})", print_console=False)
            write_log(LOG_FILE, "=" * 50, print_console=False)

            for k, v in all_metrics.items():
                result_str = f"{k.upper():12s}: {np.mean(v):.4f} (+/- {np.std(v):.4f})"
                write_log(LOG_FILE, result_str)

            # Combined ROC
            print("\n[INFO] Generating Combined ROC Curve...")
            y_pred_proba_all = np.array(y_pred_proba_all)

            plt.figure(figsize=(10, 8))
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'{class_names[1]} vs {class_names[0]} (AUC = {roc_auc:.2f})')
            else:
                y_true_all_bin = label_binarize(y_true_all, classes=range(num_classes))
                colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

                for i, color in zip(range(num_classes), colors):
                    if np.sum(y_true_all_bin[:, i]) == 0:
                        continue
                    fpr, tpr, _ = roc_curve(y_true_all_bin[:, i], y_pred_proba_all[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Total {len(all_metrics["acc"])}-Fold Combined ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(exp_result_dir, "Combined_ROC_Curve.png"), dpi=300)
            plt.close()

            # Combined Confusion Matrix
            print("\n[INFO] Generating Combined Confusion Matrix...")
            final_cm = confusion_matrix(y_true_all, y_pred_all)

            plt.figure(figsize=(8, 6))
            sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Total {len(all_metrics["acc"])}-Fold Combined CM', fontsize=15)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_result_dir, "Combined_Confusion_Matrix.png"), dpi=300)
            plt.close()

            # Summary row
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
                'acc_mean': np.mean(all_metrics['acc']),
                'acc_std': np.std(all_metrics['acc']),
                'prec_mean': np.mean(all_metrics['prec']),
                'prec_std': np.std(all_metrics['prec']),
                'rec_mean': np.mean(all_metrics['rec']),
                'rec_std': np.std(all_metrics['rec']),
                'f1_mean': np.mean(all_metrics['f1']),
                'f1_std': np.std(all_metrics['f1']),
                'spec_mean': np.mean(all_metrics['spec']),
                'spec_std': np.std(all_metrics['spec']),
                'auc_mean': np.mean(all_metrics['auc']),
                'auc_std': np.std(all_metrics['auc']),
            }
            summary_results.append(summary_row)

            summary_df = pd.DataFrame(summary_results)
            summary_df.to_csv(
                os.path.join(Config.RESULTS_DIR, "ablation_summary.csv"),
                index=False,
                encoding='utf-8-sig'
            )

    print("\n[INFO] 모든 ablation 실험이 종료되었습니다.")
from operator import sub
import re
import random

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
from itertools import cycle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer, Concatenate, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
import gc

# ============================================================
# [USER CONFIGURATION] 하이퍼 파라미터 및 설정
# ============================================================
class Config:
    # ND, CD, ED, MD 4개 클래스 설정
    FOLDER_PATH = "Distraction_dataset_Final_Merged"
    RESULTS_DIR = "Results_4Class_ND_CD_ED_MD_csv_split"

    SEED = 42

    # 원본 데이터 라벨 정의: ND=0, CD=1, ED=2, MD=3
    # Key: 원본 라벨, Value: 학습용 라벨
    TARGET_LABELS_MAP = {
        0: 0,  # ND -> Class 0
        1: 1,  # CD -> Class 1
        2: 2,  # ED -> Class 2
        3: 3   # MD -> Class 3
    }

    # 그래프 및 결과 출력용 클래스 이름 리스트
    CLASS_NAMES = ['ND', 'CD', 'ED', 'MD']

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
    EPOCHS = 50
    BATCH_SIZE = 64
    PATIENCE = 10

    # CSV 기준 분할 비율
    VAL_RATIO = 0.2


# ============================================================
# GPU 설정
# ============================================================
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


# ============================================================
# 랜덤 시드 설정
# ============================================================
def set_seeds(seed=Config.SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] 랜덤 시드 설정: {seed}")


# ============================================================
# 로그 기록용 헬퍼 함수
# ============================================================
def write_log(filepath, message, print_console=True):
    if print_console:
        print(message)

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(message + "\n")


# ============================================================
# 전체 CSV 파일 목록 수집
# ============================================================
def get_all_csv_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        return []

    csv_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ])

    if not csv_files:
        print("[ERROR] CSV 파일이 없습니다.")
        return []

    return csv_files


# ============================================================
# subject id 추출
# ============================================================
def extract_subject_id(file_path, fallback_id=-1):
    filename = os.path.basename(file_path)
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return fallback_id


# ============================================================
# 시선 운동학적 특성 계산 + baseline 보정
# 주의: 현재 baseline 보정은 Distraction 라벨을 사용하므로
# 평가 누수 가능성이 있습니다.
# ============================================================
def preprocess_subject_data(df, au_features, gaze_angle_features, pose_features):
    df = df.copy()
    sampling_rate = Config.FPS

    if 'timestamp' in df.columns:
        dt = df['timestamp'].diff()
        dt = dt.replace(0, np.nan)
        mean_dt = dt.mean() if not np.isnan(dt.mean()) else (1 / sampling_rate)
        dt = dt.fillna(mean_dt)
    else:
        dt = 1 / sampling_rate

    if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
        dx = df['gaze_angle_x'].diff()
        dy = df['gaze_angle_y'].diff()
        amp_rad = np.sqrt(dx**2 + dy**2).fillna(0)
        df['gaze_amp'] = np.degrees(amp_rad)
        df['gaze_vel'] = (df['gaze_amp'] / dt).fillna(0)
        d_vel = df['gaze_vel'].diff()
        df['gaze_acc'] = (d_vel / dt).fillna(0)
        df.loc[0:1, 'gaze_acc'] = 0
    else:
        df['gaze_amp'] = 0.0
        df['gaze_vel'] = 0.0
        df['gaze_acc'] = 0.0

    # Baseline Correction (현재 코드는 ND 정답 라벨 사용)
    nd_data = df[df['Distraction'] == 0]

    if not nd_data.empty:
        targets = au_features + gaze_angle_features + pose_features
        means = nd_data[targets].mean()
        df[targets] = df[targets] - means

    return df


# ============================================================
# CSV 목록을 입력받아 sliding window 생성
# ============================================================
def load_and_create_sliding_window_data(csv_files):
    target_time_steps = Config.TIME_STEPS

    print(f"[INFO] 타겟 클래스 매핑: {Config.TARGET_LABELS_MAP}")
    print(f"[INFO] 클래스 이름: {Config.CLASS_NAMES}")

    if not csv_files:
        print("[ERROR] 입력된 CSV 파일이 없습니다.")
        return None, None, None, None, None

    # Feature 정의
    au_features = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
        'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
        'AU25_r', 'AU26_r', 'AU45_r'
    ]
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']

    gaze_raw = [
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y'
    ]
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']
    gaze_total_features = gaze_raw + kinematic_features

    columns_to_load = au_features + pose_features + vehicle_features + gaze_raw + ['Distraction']
    final_features = au_features + pose_features + vehicle_features + gaze_total_features

    # Feature Slicing
    feature_slices = []
    current_idx = 0

    feature_slices.append((current_idx, current_idx + len(au_features)))
    current_idx += len(au_features)

    feature_slices.append((current_idx, current_idx + len(pose_features)))
    current_idx += len(pose_features)

    feature_slices.append((current_idx, current_idx + len(vehicle_features)))
    current_idx += len(vehicle_features)

    feature_slices.append((current_idx, current_idx + len(gaze_total_features)))
    current_idx += len(gaze_total_features)

    print(f"[INFO] Feature Order: AU -> Pose -> Vehicle -> Gaze (Total: {len(final_features)})")

    X_list, y_list, groups_list, subject_list = [], [], [], []

    for file_idx, file in enumerate(csv_files):
        try:
            subject_id = extract_subject_id(file, fallback_id=file_idx)

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

            # 실제 FPS 기반의 파일별 동적 윈도우/스텝 사이즈 계산
            file_window_size = int(round(actual_fps * Config.WINDOW_SECONDS))
            file_step_size = int(round(actual_fps * Config.STRIDE_SECONDS))

            if file_window_size <= 0 or file_step_size <= 0:
                print(f"[WARNING] 비정상 window/step 크기 ({file})")
                continue

            gaze_angle_cols = ['gaze_angle_x', 'gaze_angle_y']
            df = preprocess_subject_data(df, au_features, gaze_angle_cols, pose_features)

            for col in final_features:
                df[col] = df[col].astype('float32')

            values = df[final_features].values
            labels = df['Distraction'].values

            if len(df) < file_window_size:
                print(f"[WARNING] 파일 길이가 window보다 짧아 제외: {os.path.basename(file)}")
                continue

            for i in range(0, len(df) - file_window_size + 1, file_step_size):
                raw_label = labels[i + file_window_size - 1]

                if raw_label in Config.TARGET_LABELS_MAP:
                    final_label = Config.TARGET_LABELS_MAP[raw_label]

                    X_window = values[i: i + file_window_size]

                    if len(X_window) < target_time_steps:
                        pad_width = target_time_steps - len(X_window)
                        X_window = np.pad(X_window, ((0, pad_width), (0, 0)), mode='edge')
                    elif len(X_window) > target_time_steps:
                        X_window = X_window[:target_time_steps]

                    X_list.append(X_window)
                    y_list.append(final_label)
                    groups_list.append(subject_id)
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


# ============================================================
# 배치 단위로 데이터를 모델에 공급
# ============================================================
class DataGenerator(Sequence):
    def __init__(self, X_data, y_data, indices, batch_size, scaler=None, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.scaler = scaler
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]

        X_batch = self.X_data[batch_indices]
        y_batch = self.y_data[batch_indices]

        if self.scaler is not None:
            b, t, f = X_batch.shape
            X_batch = self.scaler.transform(X_batch.reshape(-1, f)).reshape(b, t, f)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# 모델 구성
# ============================================================
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        hidden_units = max(channel // self.ratio, 1)

        self.shared_layer_one = Dense(
            hidden_units,
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
        x_slice = Lambda(
            lambda x, s=start, e=end: x[:, :, s:e],
            name=f'modality_slice_{i}'
        )(inputs)

        x = Conv1D(
            filters=Config.CNN_FILTERS,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name=f'conv1_{i}'
        )(x_slice)

        x = Conv1D(
            filters=Config.CNN_FILTERS,
            kernel_size=3,
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
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 학습 곡선 시각화
# ============================================================
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


# ============================================================
# 평가 함수
# ============================================================
def evaluate_fold(model, val_gen, class_names, fold_no, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    y_true = []
    for i in range(len(val_gen)):
        _, batch_y = val_gen[i]
        y_true.extend(batch_y)
    y_true = np.array(y_true)

    y_pred = y_pred[:len(y_true)]
    y_pred_proba = y_pred_proba[:len(y_true)]

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
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
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = TN / (TN + FP)
        class_specificity = np.nan_to_num(class_specificity)

    spec = np.mean(class_specificity)

    log_msg = (
        f"  [Result Fold {fold_no}] "
        f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
        f"F1: {f1:.4f}, Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    )
    write_log(log_file, log_msg)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f'Fold {fold_no} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"CM_Fold_{fold_no}.png"), dpi=300)
    plt.close()

    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'spec': spec,
        'auc': auc_score
    }, y_true, y_pred, y_pred_proba


# ============================================================
# 클래스 밸런싱 (train에만 적용)
# ============================================================
def balancing_classes(X, y, subjects, num_classes):
    X_balanced_list, y_balanced_list, subjects_balanced_list = [], [], []

    for sub in sorted(np.unique(subjects)):
        idx = np.where(subjects == sub)[0]
        X_sub = X[idx]
        y_sub = y[idx]
        subjects_sub = subjects[idx]

        class_indices = [np.where(y_sub == i)[0] for i in range(num_classes)]
        valid_lengths = [len(arr) for arr in class_indices if len(arr) > 0]
        min_size = min(valid_lengths, default=0)

        if min_size == 0:
            print(f"[WARNING] Subject {sub}는 일부 클래스가 없어 balancing에서 제외됩니다.")
            continue

        selected_indices = []
        for arr in class_indices:
            if len(arr) > 0:
                selected = np.random.choice(arr, min_size, replace=False)
                selected_indices.append(selected)

        balanced_indices = np.concatenate(selected_indices)
        balanced_indices.sort()

        X_balanced = X_sub[balanced_indices]
        y_balanced = y_sub[balanced_indices]
        subjects_balanced = subjects_sub[balanced_indices]

        y_sub_sum = [np.sum(y_sub == i) for i in range(num_classes)]
        y_balanced_sum = [np.sum(y_balanced == i) for i in range(num_classes)]
        print(f"Subject {sub} - {y_sub_sum} → {y_balanced_sum}")

        X_balanced_list.append(X_balanced)
        y_balanced_list.append(y_balanced)
        subjects_balanced_list.append(subjects_balanced)

    if not X_balanced_list:
        return None, None, None

    X_new = np.concatenate(X_balanced_list, axis=0)
    y_new = np.concatenate(y_balanced_list, axis=0)
    subjects_new = np.concatenate(subjects_balanced_list, axis=0)

    return X_new, y_new, subjects_new


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    set_seeds()

    num_classes = len(Config.TARGET_LABELS_MAP)

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    LOG_FILE = os.path.join(Config.RESULTS_DIR, "training_result_log.txt")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Start: {Config.CLASS_NAMES}\n")
        f.write("=" * 50 + "\n")

    print("=" * 80)
    print(f"🚀 Multi-Class Classification: {Config.CLASS_NAMES}")
    print("=" * 80)

    # 1) 전체 CSV 수집
    all_csv_files = get_all_csv_files(Config.FOLDER_PATH)

    if len(all_csv_files) < 2:
        print("[ERROR] train/val 분할을 위해 최소 2개 이상의 CSV가 필요합니다.")
        raise SystemExit

    # 2) CSV 기준으로 train / val 분리
    train_files, val_files = train_test_split(
        all_csv_files,
        test_size=Config.VAL_RATIO,
        random_state=Config.SEED,
        shuffle=True
    )

    print("\n[INFO] Train CSV Files")
    for f in train_files:
        print(" ", os.path.basename(f))

    print("\n[INFO] Validation CSV Files")
    for f in val_files:
        print(" ", os.path.basename(f))

    # 3) train CSV로만 train window 생성
    X_train, y_train, groups_train, subjects_train, feature_slices = load_and_create_sliding_window_data(train_files)

    # 4) val CSV로만 val window 생성
    X_val, y_val, groups_val, subjects_val, _ = load_and_create_sliding_window_data(val_files)

    if X_train is None or X_val is None:
        print("학습 또는 검증 데이터를 생성하지 못했습니다.")
        raise SystemExit

    print(f"\n[Train Data] Input Shape: {X_train.shape}")
    print(f"[Val Data]   Input Shape: {X_val.shape}")

    print(f"\n[INFO] Train Labels Distribution Before Balancing")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    # 5) balancing은 train에만 적용
    X_train, y_train, subjects_train = balancing_classes(X_train, y_train, subjects_train, num_classes)

    if X_train is None:
        print("[ERROR] balancing 후 학습 데이터가 없습니다.")
        raise SystemExit

    print(f"\n[INFO] Train Labels Distribution After Balancing")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    print(f"\n[INFO] Validation Labels Distribution")
    unique, counts = np.unique(y_val, return_counts=True)
    print(dict(zip(unique, counts)))

    # 6) scaler는 train에만 fit
    scaler = MinMaxScaler()
    try:
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_flat)
        del X_train_flat
        gc.collect()
    except MemoryError:
        print("[WARNING] Scaler fit 중 메모리 부족. 샘플링하여 fit 진행.")
        sample_size = max(1, len(X_train) // 10)
        sample_idx = np.random.choice(np.arange(len(X_train)), size=sample_size, replace=False)
        X_sample_flat = X_train[sample_idx].reshape(-1, X_train.shape[-1])
        scaler.fit(X_sample_flat)
        del X_sample_flat
        gc.collect()

    # 7) generator 생성
    train_gen = DataGenerator(
        X_train, y_train, np.arange(len(X_train)),
        Config.BATCH_SIZE, scaler=scaler, shuffle=True
    )

    val_gen = DataGenerator(
        X_val, y_val, np.arange(len(X_val)),
        Config.BATCH_SIZE, scaler=scaler, shuffle=False
    )

    # 8) 모델 학습
    model = build_model((X_train.shape[1], X_train.shape[2]), feature_slices, num_classes)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=Config.PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        epochs=Config.EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stopping],
        verbose=1
    )

    plot_learning_curve(history, 1, Config.RESULTS_DIR)

    # 9) 평가
    result, y_t, y_p, y_proba = evaluate_fold(
        model, val_gen, Config.CLASS_NAMES, 1, Config.RESULTS_DIR, LOG_FILE
    )

    print("\n" + "=" * 50)
    print("Final Hold-out Validation Performance")
    print("=" * 50)
    for k, v in result.items():
        print(f"{k.upper():12s}: {v:.4f}")

    # 10) 최종 confusion matrix 저장
    final_cm = confusion_matrix(y_t, y_p, labels=np.arange(num_classes))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        final_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES
    )
    plt.title("Validation Confusion Matrix", fontsize=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "Validation_Confusion_Matrix.png"), dpi=300)
    plt.show()
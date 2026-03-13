from operator import sub
import re 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from itertools import cycle 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer, Concatenate, Lambda
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
import gc

# [USER CONFIGURATION] 하이퍼 파라미터 및 설정

class Config:
    # ND, CD, ED, MD 4개 클래스 설정
    FOLDER_PATH = "Distraction_dataset_Final_Merged"              # 경로 및 시드 설정(데이터 폴더 경로)
    RESULTS_DIR = "Results_4Class_ND_CD_ED_MD_260311_test"        # 결과 저장 경로

    SEED = 42

    # 원본 데이터 라벨 정의: ND=0, CD=1, ED=2, MD=3
    # Key: 원본 라벨, Value: 학습용 라벨
    TARGET_LABELS_MAP = {
        0: 0,  # ND -> Class 0
        1: 1,  # CD -> Class 1
        2: 2,  # ED -> Class 2
        3: 3   # MD -> Class 3
    }
    
    # 그래프 및 결과 출력용 클래스 이름 리스트 (위 Value 순서와 일치해야 함)
    CLASS_NAMES = ['ND', 'CD', 'ED', 'MD']       

    # 3. 데이터 전처리 (Sliding Window)
    FPS = 28
    WINDOW_SECONDS = 10
    STRIDE_SECONDS = 5
    TIME_STEPS = FPS * WINDOW_SECONDS
    STEP_SIZE = FPS * STRIDE_SECONDS

    # 4. 모델 구조 파라미터
    CNN_FILTERS = 32                # CNN 필터 개수
    CNN_KERNEL_SIZE = 3             # 커널 크기
    LSTM_UNITS = 64                 # LSTM 유닛 개수
    DENSE_UNITS = 32                # FC Layer 유닛 개수
    DROPOUT_RATE = 0.4              # Dropout 비율

    # 5. 학습 파라미터
    N_SPLITS = 5                    # K-Fold 분할 수
    EPOCHS = 100                    # 학습 에폭 수
    BATCH_SIZE = 64                 # 배치 크기
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# GPU 설정
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

# 랜덤 시드 설정 함수
def set_seeds(seed=Config.SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] 랜덤 시드 설정: {seed}")

# 로그 기록용 헬퍼 함수(파일에 메시지 기록 + 콘솔 출력)
def write_log(filepath, message, print_console=True):
    if print_console:
        print(message)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

# 1. 시선 운동학적 특성(Kinematics) 계산 함수 (및 통합 Baseline 보정)
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

    # Baseline Correction (ND 기준) - AU 피처에만 적용 (-> Gaze, Pose 통합 적용으로 개선)
    nd_data = df[df['Distraction'] == 0]
    
    if not nd_data.empty:
        targets = au_features + gaze_angle_features + pose_features
        means = nd_data[targets].mean()
        df[targets] = df[targets] - means

    return df

# 2. 데이터 로드 (Config 맵핑 기반 로드 & 동적 FPS 패딩 적용)
def load_and_create_sliding_window_data():
    folder_path = Config.FOLDER_PATH
    target_time_steps = Config.TIME_STEPS # 목표 윈도우 크기 (고정 280)
    
    print(f"[INFO] 타겟 클래스 매핑: {Config.TARGET_LABELS_MAP}")
    print(f"[INFO] 클래스 이름: {Config.CLASS_NAMES}")

    # 폴더 및 파일 존재 여부 확인
    if not os.path.isdir(folder_path):
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {folder_path}")
        return None, None, None, None, None

    csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])
    if not csv_files:
        print(f"[ERROR] CSV 파일이 없습니다.")
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

    # 현민 수정
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

            # 실제 FPS 기반의 파일별 동적 윈도우/스텝 사이즈 계산
            file_window_size = int(round(actual_fps * Config.WINDOW_SECONDS))
            file_step_size = int(round(actual_fps * Config.STRIDE_SECONDS))

            # 통합 전처리 (calculate_gaze_kinematics + Baseline 보정)
            gaze_angle_cols = ['gaze_angle_x', 'gaze_angle_y']
            df = preprocess_subject_data(df, au_features, gaze_angle_cols, pose_features)
            
            for col in final_features:
                df[col] = df[col].astype('float32')

            values = df[final_features].values
            labels = df['Distraction'].values

            # 동적으로 계산된 파일별 스텝 사이즈 적용
            for i in range(0, len(df) - file_window_size + 1, file_step_size):
                raw_label = labels[i + file_window_size - 1]
                
                # Config에 정의된 매핑 사용
                if raw_label in Config.TARGET_LABELS_MAP:
                    final_label = Config.TARGET_LABELS_MAP[raw_label]
                    
                    # 파일의 실제 FPS에 맞춰 추출
                    X_window = values[i : i + file_window_size]
                    
                    # 고정 크기(280)로 Padding 또는 Truncating
                    if len(X_window) < target_time_steps:
                        pad_width = target_time_steps - len(X_window)
                        # 마지막 값을 부족한 프레임만큼 반복해서 덧붙임
                        X_window = np.pad(X_window, ((0, pad_width), (0, 0)), mode='edge')
                    elif len(X_window) > target_time_steps:
                        # 지정된 크기보다 길면 잘라냄
                        X_window = X_window[:target_time_steps]
                    
                    X_list.append(X_window)
                    y_list.append(final_label)
                    groups_list.append(subject_id)
                    
                    # 현민 수정
                    filename = os.path.basename(file)
                    match = re.search(r'T(\d+)-(\d+)', filename)

                    if match:
                        person_id = int(match.group(1))   # T001 -> 1
                        state_id = int(match.group(2))    # 005 -> 5 (지금은 안 써도 됨)
                        subject_list.append(person_id)
                    else:
                        print(f"[WARNING] 파일명 형식을 파싱하지 못했습니다: {filename}")
                        subject_list.append(subject_id)
                else:
                    # 매핑에 없는 라벨은 제외
                    continue

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

# 배치 단위로 데이터를 모델에 공급
class DataGenerator(Sequence):
    def __init__(self, X_data, y_data, indices, batch_size, scaler=None, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.indices = indices
        self.batch_size = batch_size
        self.scaler = scaler
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch = self.X_data[batch_indices]
        y_batch = self.y_data[batch_indices]

        if self.scaler:
            b, t, f = X_batch.shape
            X_batch = self.scaler.transform(X_batch.reshape(-1, f)).reshape(b, t, f)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# 3. 모델 구성
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(channel // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
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
        self.W = self.add_weight(name='att_weight', shape=(feature_dim, feature_dim),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(feature_dim,),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(feature_dim, 1),
                                 initializer='glorot_uniform', trainable=True)
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
    
    # 각 모달리티별로 독립적인 CNN-LSTM 경로 생성 (Late Fusion)
    for i, (start, end) in enumerate(feature_slices):
        # 1. 데이터 슬라이싱
        x_slice = Lambda(lambda x, s=start, e=end: x[:, :, s:e], name=f'modality_slice_{i}')(inputs)
        
        # 2. 독립적인 특징 추출 (1D-CNN 블록)
        x = Conv1D(filters=Config.CNN_FILTERS, kernel_size=3, padding='same', 
                   activation='relu', name=f'conv1_{i}')(x_slice)
        x = MaxPooling1D(pool_size=2, name=f'pool1_{i}')(x)
        x = Dropout(Config.DROPOUT_RATE, name=f'drop1_{i}')(x)
        
        x = Conv1D(filters=Config.CNN_FILTERS * 2, kernel_size=3, padding='same', 
                   activation='relu', name=f'conv2_{i}')(x)
        x = MaxPooling1D(pool_size=2, name=f'pool2_{i}')(x)
        x = Dropout(Config.DROPOUT_RATE, name=f'drop2_{i}')(x)

        # 3. 채널 어텐션 (사용자 기존 코드 유지)
        x = ChannelAttention(ratio=8, name=f'channel_att_{i}')(x)
        
        # 4. 모달리티별 독립적 시계열 학습 (LSTM)
        # 각 모달리티의 고유한 시간적 흐름을 보존합니다.
        x = LSTM(units=Config.LSTM_UNITS, return_sequences=True, name=f'lstm_{i}')(x)
        encoded_branches.append(x)
    
    # 5. 특징 융합 (Late Fusion Step)
    # 모든 모달리티의 은닉 상태(Hidden States)를 합칩니다.
    if len(encoded_branches) > 1:
        x = Concatenate(axis=-1, name='modality_concat')(encoded_branches)
    else:
        x = encoded_branches[0]
        
    # 6. 셀프 어텐션 기반 가중치 할당
    # 융합된 특징 중 스트레스 감지에 중요한 정보에 집중합니다.
    x = SoftAttention(name='soft_attention')(x) 
    
    # 7. 최종 분류
    x = Dense(Config.DENSE_UNITS, activation='relu', name='dense_fc')(x)
    x = Dropout(Config.DROPOUT_RATE, name='dense_dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ==================
# 학습 곡선(Learning Curve) 시각화 함수 추가
# ==================
def plot_learning_curve(history, fold_no, save_dir):
    """
    Epoch별 Train/Validation Accuracy 및 Loss를 시각화하여 저장합니다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2)
    axes[0].set_title(f'Fold {fold_no} - Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
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

# ==================
# 4. 평가 및 메인 실행
# ==================
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

    # --- ROC Curve (Multi-Class) ---
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    # [설명] 색상이 4개이므로 ND, CD, ED, MD 각각 다른 색상으로 표현됩니다.
    colors = cycle(['blue', 'red', 'green', 'orange']) 
    
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_true_bin[:, i]) == 0:
            print(f"[WARNING] Class {class_names[i]} has no positive samples in Fold {fold_no}.")
            continue
            
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

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

    # --- Metrics Calculation (Specificity & AUC 추가) ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_score = 0.0

    # Specificity Calculation
    cm = confusion_matrix(y_true, y_pred)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = TN / (TN + FP)
        class_specificity = np.nan_to_num(class_specificity) 
        
    spec = np.mean(class_specificity) 

    log_msg = f"  [Result Fold {fold_no}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Spec: {spec:.4f}, AUC: {auc_score:.4f}"
    write_log(log_file, log_msg)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Fold {fold_no} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"CM_Fold_{fold_no}.png"))
    plt.close()

    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'spec': spec, 'auc': auc_score
    }, y_true, y_pred, y_pred_proba

if __name__ == "__main__":
    set_seeds()
    
    # Config에서 정의한 클래스 개수 가져오기
    num_classes = len(Config.TARGET_LABELS_MAP)
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(Config.RESULTS_DIR, "training_result_log.txt")

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Start: {Config.CLASS_NAMES}\n")
        f.write("="*50 + "\n")

    print("="*80)
    print(f"🚀 Multi-Class Classification (Subject-Independent): {Config.CLASS_NAMES}")
    print("="*80)

    # 데이터 로드
    X, y, groups, subjects, feature_slices = load_and_create_sliding_window_data()
    
    if X is not None:
        """
        현민
        클래스 밸런싱 추가 및 StratifiedGroupKFold 적용을 위한 수정
        """
        def balancing_classes(X, y, subjects:np.array):
            X_balanced_list, y_balanced_list, subjects_balanced_list = [], [], []
            for sub in sorted(np.unique(subjects)):
                idx = np.where(subjects == sub)[0]
                X_sub = X[idx]
                y_sub = y[idx]
                sub_sub = subjects[idx] # 피험자 정보 보존
                
                # 1. 각 클래스별(0, 1, 2, 3) 인덱스 찾기
                indices_0 = np.where(y_sub == 0)[0]
                indices_1 = np.where(y_sub == 1)[0]
                indices_2 = np.where(y_sub == 2)[0]
                indices_3 = np.where(y_sub == 3)[0]

                # 2. 클래스 중 가장 적은 샘플 개수 확인
                min_size = min((len(i) for i in [indices_0, indices_1, indices_2, indices_3] if len(i) > 0), default=0)    

                # 3. 각 클래스에서 min_size만큼 무작위 샘플링
                selected_0 = np.random.choice(indices_0, min_size, replace=False) if len(indices_0) > 0 else indices_0
                selected_1 = np.random.choice(indices_1, min_size, replace=False) if len(indices_1) > 0 else indices_1
                selected_2 = np.random.choice(indices_2, min_size, replace=False) if len(indices_2) > 0 else indices_2
                selected_3 = np.random.choice(indices_3, min_size, replace=False) if len(indices_3) > 0 else indices_3

                # 4. 선택된 인덱스 합치기 및 정렬
                balanced_indices = np.concatenate([selected_0, selected_1, selected_2, selected_3])
                balanced_indices.sort()

                # 5. 최종 데이터 추출
                X_balanced = X_sub[balanced_indices]
                y_balanced = y_sub[balanced_indices]
                subjects_balanced = sub_sub[balanced_indices] # 피험자 배열 추출 추가
                
                y_sub_sum = [sum(y_sub == i) for i in range(num_classes)]
                y_balanced_sum = [sum(y_balanced == i) for i in range(num_classes)]
                print(f"Subject {sub} - {y_sub_sum} → {y_balanced_sum}")
                
                X_balanced_list.append(X_balanced)
                y_balanced_list.append(y_balanced)
                subjects_balanced_list.append(subjects_balanced) # 피험자 리스트 추가
                
            X_new = np.concatenate(X_balanced_list, axis=0)
            y_new = np.concatenate(y_balanced_list, axis=0)
            subjects_new = np.concatenate(subjects_balanced_list, axis=0) # 최종 병합 반환
            
            return X_new, y_new, subjects_new
            
        print(f"\n[INFO] Applying Global Balancing...")
        print(f"Before Balancing: Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # subjects 반환값을 받도록 수정
        X, y, subjects = balancing_classes(X, y, subjects)
        
        print(f"After Balancing: Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Total Samples: {X.shape[0]}, Subjects: {len(np.unique(subjects))}")
        print("="*50)

        print(f"\n[Data Info] Input Shape: {X.shape}")
        
        # 현민: StratifiedKFold 대신 StratifiedGroupKFold 적용 (CSV 단위, 피험자 단위 분할)
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED) 
        
        all_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'spec': [], 'auc': []}
        
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        
        fold_no = 1
        
        for train_idx, val_idx in sgkf.split(X, y, groups=subjects): 
            print(f"\nTraining Fold {fold_no}...")

            y_train_fold = y[train_idx]
            unique_classes, class_counts = np.unique(y_train_fold, return_counts=True)
            min_count = class_counts.min()
            
            print(f"  > Class Balance Adjustment: Target count per class = {min_count}")
            
            balanced_train_indices = []
            for cls in unique_classes:
                cls_indices = train_idx[y[train_idx] == cls]
                selected_indices = np.random.choice(cls_indices, min_count, replace=False)
                balanced_train_indices.extend(selected_indices)
            
            balanced_train_idx = np.array(balanced_train_indices)
            np.random.shuffle(balanced_train_idx)
            
            scaler = MinMaxScaler()
            try:
                X_train_flat = X[balanced_train_idx].reshape(-1, X.shape[-1])
                scaler.fit(X_train_flat)
                del X_train_flat
                gc.collect()
            except MemoryError:
                print("[WARNING] Scaler fit 중 메모리 부족. 샘플링하여 fit 진행.")
                sample_idx = np.random.choice(balanced_train_idx, size=len(balanced_train_idx)//10, replace=False)
                X_sample_flat = X[sample_idx].reshape(-1, X.shape[-1])
                scaler.fit(X_sample_flat)
                del X_sample_flat
                gc.collect()

            train_gen = DataGenerator(X, y, balanced_train_idx, Config.BATCH_SIZE, scaler=scaler, shuffle=True)
            val_gen = DataGenerator(X, y, val_idx, Config.BATCH_SIZE, scaler=scaler, shuffle=False)

            model = build_model((X.shape[1], X.shape[2]), feature_slices, num_classes)

            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR,
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

            plot_learning_curve(history, fold_no, Config.RESULTS_DIR)

            fold_res, y_t, y_p, y_proba = evaluate_fold(model, val_gen, Config.CLASS_NAMES, fold_no, Config.RESULTS_DIR, LOG_FILE)
            
            for k in all_metrics:
                all_metrics[k].append(fold_res[k])
            
            y_true_all.extend(y_t)
            y_pred_all.extend(y_p)
            y_pred_proba_all.extend(y_proba)

            del model, train_gen, val_gen, scaler
            K.clear_session()
            gc.collect()
            fold_no += 1

        print("\n" + "="*50)
        print(f"Final 5-Fold Average Performance ({Config.CLASS_NAMES})")
        print("="*50)
        
        write_log(LOG_FILE, "\n" + "="*50, print_console=False)
        write_log(LOG_FILE, f"Final 5-Fold Average Performance ({Config.CLASS_NAMES})", print_console=False)
        write_log(LOG_FILE, "="*50, print_console=False)

        for k, v in all_metrics.items():
            result_str = f"{k.upper():12s}: {np.mean(v):.4f} (+/- {np.std(v):.4f})"
            write_log(LOG_FILE, result_str)
        
        # --- Combined ROC ---
        print("\n[INFO] Generating Combined ROC Curve...")
        y_true_all_bin = label_binarize(y_true_all, classes=range(num_classes))
        y_pred_proba_all = np.array(y_pred_proba_all)

        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange'])

        for i, color in zip(range(num_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_all_bin[:, i], y_pred_proba_all[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{Config.CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Total 5-Fold Combined ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(Config.RESULTS_DIR, "Combined_ROC_Curve.png"), dpi=300)
        plt.close()

        print("\n[INFO] Generating Combined Confusion Matrix...")
        final_cm = confusion_matrix(y_true_all, y_pred_all)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
        plt.title(f'Total 5-Fold Combined CM', fontsize=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, "Combined_Confusion_Matrix.png"), dpi=300)
        plt.show()

    else:
        print("학습을 진행할 데이터가 충분하지 않습니다.")
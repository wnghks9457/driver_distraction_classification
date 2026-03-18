import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 구성
data = {
    'Model': [],
    'Task': [],
    'Metric': [],
    'Value': []
}

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
tasks = ['ND/CD/MD', 'ND/ED/MD', 'ND/CDED/MD', 'ND/CD/ED/MD']

# 모델별 측정값 (순서: ND/CD/MD, ND/ED/MD, ND/CDED/MD, ND/CD/ED/MD)
# 실험 후 빈칸(np.nan)에 값을 채워넣으시면 됩니다.
raw_data = {
    'KNN': [
        [0.6115, 0.6032, 0.6546, 0.4238], # Accuracy
        [0.6715, 0.6600, 0.7041, 0.4478], # Precision
        [0.6115, 0.6032, 0.6546, 0.4238], # Recall
        [0.6036, 0.5960, 0.6508, 0.4098], # F1-Score
        [0.8057, 0.8016, 0.8273, 0.8079], # Specificity
        [0.8078, 0.8021, 0.8366, 0.6748]  # AUC
    ],
    'RF': [
        [0.7581, 0.7397, 0.7982, 0.5388],
        [0.7817, 0.7763, 0.8088, 0.5382],
        [0.7581, 0.7397, 0.7982, 0.5388],
        [0.7577, 0.7383, 0.7988, 0.5288],
        [0.8791, 0.8699, 0.8991, 0.8463],
        [0.9177, 0.9203, 0.9317, 0.8049]
    ],
    'SVM': [
        [0.7497, 0.7354, 0.7561, 0.5358],
        [0.7573, 0.7421, 0.7604, 0.5186],
        [0.7497, 0.7354, 0.7561, 0.5358],
        [0.7492, 0.7346, 0.7565, 0.5219],
        [0.8749, 0.8677, 0.8780, 0.8453],
        [0.8993, 0.8793, 0.8986, 0.7967]
    ],
    'XGBoost': [
        [0.7583, 0.7275, 0.7767, 0.5539], 
        [0.7977, 0.7819, 0.7987, 0.5589],
        [0.7583, 0.7275, 0.7767, 0.5539],
        [0.7587, 0.7259, 0.7760, 0.5408],
        [0.8792, 0.8638, 0.8884, 0.8513],
        [0.9205, 0.9220, 0.9318, 0.8163]
    ],
    'CNN-LSTM': [
        [0.8093, 0.8057, 0.7817, 0.6337],
        [0.8158, 0.8123, 0.7894, 0.6297],
        [0.8119, 0.8081, 0.7777, 0.6125],
        [0.8115, 0.8079, 0.7795, 0.6077],
        [0.9019, 0.8982, 0.8858, 0.8769],
        [0.9234, 0.9201, 0.9092, 0.8588]
    ]
}

# 데이터를 Pandas DataFrame 형식으로 변환
for model, metric_values in raw_data.items():
    for i, metric in enumerate(metrics):
        for j, task in enumerate(tasks):
            data['Model'].append(model)
            data['Task'].append(task)
            data['Metric'].append(metric)
            data['Value'].append(metric_values[i][j])

df = pd.DataFrame(data)

# 2. 그래프 그리기 (Seaborn)
sns.set_theme(style="whitegrid")

# 각 Task별로 그래프를 개별적으로 생성
for task in tasks:
    # 개별 캔버스 생성 (가로로 긴 비율로 설정하여 텍스트 겹침 방지)
    plt.figure(figsize=(14, 6))
    
    # 해당 Task 데이터 필터링
    subset = df[df['Task'] == task]
    
    # Barplot 생성
    ax = sns.barplot(
        data=subset, 
        x='Metric', 
        y='Value', 
        hue='Model', 
        palette='Set2'
    )
    
    plt.title(f'Class Classification: {task}', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Metric', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    # 수치 표기를 위해 y축 상단 여유 확보
    plt.ylim(0, 1.15) 
    
    # 각 막대 위에 수치 표시 (가로 방향 표기)
    for p in ax.patches:
        height = p.get_height()
        # np.nan이 아니고 높이가 0보다 클 때만 표기
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.4f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=10, color='black', xytext=(0, 4), 
                        textcoords='offset points') # rotation=90 제거됨

    # 범례 위치 조정 (그래프 우측 바깥쪽으로 배치)
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

    plt.tight_layout()
    plt.show() # 개별 그래프 출력
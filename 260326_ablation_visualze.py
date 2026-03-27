import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('ablation_summary_knn.csv')

# 2. 분석 대상 데이터 필터링 (모든 실험의 baseline 및 4가지 주요 특성 그룹 데이터 추출)
target_names = ['none', 'AU', 'POSE', 'VEHICLE', 'GAZE']
plot_df = df[df['removed_name'].isin(target_names)].copy()

# 3. 데이터 전처리 및 이름 수정
# x축에 표시할 이름을 요청하신 대로 매핑하여 변경합니다.
rename_map = {
    'none': 'Baseline',
    'AU': 'AU',
    'POSE': 'Head Pose',
    'VEHICLE': 'Vehicle',
    'GAZE': 'Gaze'
}
plot_df['removed_name'] = plot_df['removed_name'].replace(rename_map)

# X축에 표시할 그룹 순서를 수정된 이름에 맞춰 고정합니다.
x_order = ['Baseline', 'AU', 'Head Pose', 'Vehicle', 'Gaze']
plot_df['removed_name'] = pd.Categorical(plot_df['removed_name'], categories=x_order, ordered=True)

# 실험 종류별로 색상을 묶기 위해 실험 이름 순서 지정 (4-Class 먼저, 그 다음 3-Class들)
exp_order = ['4Class_ND_CD_ED_MD', '3Class_ND_ED_MD', '3Class_ND_CDED_MD', '3Class_ND_CD_MD']
plot_df['experiment'] = pd.Categorical(plot_df['experiment'], categories=exp_order, ordered=True)

# 데이터 정렬
plot_df = plot_df.sort_values(['removed_name', 'experiment'])

# 4. 시각화 설정 
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")

# x축: 그룹 이름, y축: 정확도(acc_mean), 색상(hue): 실험 종류
ax = sns.barplot(data=plot_df, x='removed_name', y='acc_mean', hue='experiment', palette='pastel')

# y축 범위를 0.0 ~ 1.0으로 고정
plt.ylim(0.0, 1.0)

# 5. 막대 위에 수치 텍스트 표시 (폰트 크기 9, 굵게)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, fontweight='bold')

# 6. 제목 및 축 라벨 설정 (굵게 처리: fontweight='bold')
plt.title('KNN Ablation Study (Accuracy Comparison)', fontsize=16, fontweight='bold')
plt.xlabel('Group', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')

# 범례 설정 (실험 종류별 색상 안내)
plt.legend(title='Experiment (Color)', bbox_to_anchor=(1.02, 1), loc='upper left', title_fontproperties={'weight': 'bold'})

# 레이아웃 자동 조정 및 출력/저장
plt.tight_layout()
plt.show() # 화면 출력
plt.savefig('accuracy_all_experiments_final.png', dpi=300, bbox_inches='tight') # 저장 시 주석 해제
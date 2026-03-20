"""
===============================================================================
운전자 주의 분산(Driver Distraction) 데이터 정규성 검정 (Shapiro-Wilk Test)
===============================================================================

[주요 기능]
- 실험 조건: CD(인지)와 ED(감정)를 하나의 통합 그룹(CD_ED)으로 묶어 분석.
- 피험자 배정: 전체 피험자 풀에서 ND, CD_ED, MD 3개 그룹으로 10회 무작위 분배.
- 분석 내용: 각 피처가 그룹별(ND, CD_ED, MD)로 정규 분포를 따르는지 Shapiro-Wilk 검정 수행.
- 결과 저장: W-statistic, P-value, 정규성 만족 여부를 CSV로 저장하고, 
  10회 시뮬레이션 종합 요약본 및 바 그래프 시각화 자료를 'Shapiro_Simulation_Results_nd-cded-md' 폴더에 저장합니다.
===============================================================================
"""

import os
import re
import random
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt  # 시각화를 위해 추가

# ==========================================
# [USER CONFIGURATION] 고정 폴더 설정
# ==========================================
class Config:
    FOLDER_PATH = "Distraction_dataset_Final_Merged_ANOVA"  # CD, ED, MD 폴더
    ND_FOLDER_PATH = "dataset_nd_merged"                    # ND 폴더
    FPS = 28
    
    # 정규성 검정을 위한 단일 마스터 폴더
    MASTER_DIR = "Shapiro_Simulation_Results_nd-cded-md"

def prepare_iteration_dirs(iteration):
    """매 회차마다 폴더와 저장 경로를 생성하여 반환합니다."""
    iter_dir = os.path.join(Config.MASTER_DIR, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    return {
        "iter_dir": iter_dir,
        "shapiro_csv": os.path.join(iter_dir, f"Feature_Shapiro_Wilk_Results_nd-cded-md_iter{iteration}.csv")
    }

def add_gaze_kinematic_features(df):
    df = df.copy()
    if 'timestamp' in df.columns:
        dt = df['timestamp'].diff().replace(0, np.nan)
        mean_dt = dt.mean() if not np.isnan(dt.mean()) else (1 / Config.FPS)
        dt = dt.fillna(mean_dt)
    else:
        dt = pd.Series([1 / Config.FPS] * len(df), index=df.index)

    if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
        dx = df['gaze_angle_x'].diff()
        dy = df['gaze_angle_y'].diff()
        amp_rad = np.sqrt(dx**2 + dy**2).fillna(0)
        df['gaze_amp'] = np.degrees(amp_rad)
        df['gaze_vel'] = (df['gaze_amp'] / dt).replace([np.inf, -np.inf], np.nan).fillna(0)
        d_vel = df['gaze_vel'].diff()
        df['gaze_acc'] = (d_vel / dt).replace([np.inf, -np.inf], np.nan).fillna(0)
        if len(df) > 0:
            df.loc[df.index[0], 'gaze_acc'] = 0
        if len(df) > 1:
            df.loc[df.index[1], 'gaze_acc'] = 0
    else:
        df['gaze_amp'] = 0.0
        df['gaze_vel'] = 0.0
        df['gaze_acc'] = 0.0
        
    return df

def summarize_csv_segment(df_segment, target_features, file_name, group_label):
    if df_segment.empty:
        return None

    summary = {
        'Source_File': file_name,
        'Group': group_label,
        'Num_Frames': len(df_segment)
    }

    for feature in target_features:
        if feature not in df_segment.columns:
            summary[feature] = np.nan
            continue
        summary[feature] = df_segment[feature].mean()

    return summary

def run_shapiro_wilk(target_features, group_sets):
    """각 비교 그룹셋의 피처별 정규성 검정(Shapiro-Wilk)을 수행합니다."""
    shapiro_dfs = {}

    for comp_name, group_dict in group_sets.items():
        shapiro_results = []
        
        for feature in target_features:
            merged_rows = []
            valid_group_names = []

            # 데이터를 Long format으로 변환
            for group_name, group_df in group_dict.items():
                if group_df.empty or feature not in group_df.columns:
                    continue
                values = group_df[feature].dropna()
                
                # Shapiro-Wilk 검정은 최소 3개의 샘플이 필요함
                if len(values) >= 3:
                    valid_group_names.append(group_name)
                    for v in values:
                        merged_rows.append({"FeatureValue": v, "Group": group_name})

            if len(valid_group_names) == 0:
                continue

            long_df = pd.DataFrame(merged_rows)
            if long_df.empty:
                continue

            try:
                # pingouin을 이용한 정규성 검정 (group별로 수행)
                norm_test = pg.normality(data=long_df, dv="FeatureValue", group="Group", method="shapiro")
                
                for group_name in norm_test.index:
                    w_stat = norm_test.loc[group_name, 'W']
                    p_val = norm_test.loc[group_name, 'pval']
                    is_normal = norm_test.loc[group_name, 'normal'] # True면 정규성 만족(p >= 0.05)
                    
                    decision = "정규성 만족" if is_normal else "정규성 위배"
                    
                    shapiro_results.append({
                        "Feature": feature,
                        "Comparison": comp_name,
                        "Group": group_name,
                        "W_statistic": round(w_stat, 4) if pd.notna(w_stat) else np.nan,
                        "P_value": p_val,
                        "Is_Normal": is_normal,
                        "Result": decision
                    })

            except Exception as e:
                pass 

        shapiro_dfs[comp_name] = pd.DataFrame(shapiro_results)

    return shapiro_dfs


def extract_all_available_subjects():
    subjects = set()
    for folder in [Config.ND_FOLDER_PATH, Config.FOLDER_PATH]:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            for f in files:
                match = re.search(r'(T\d{3})', f)
                if match:
                    subjects.add(match.group(1))
    return sorted(list(subjects))


def test_features_normality_csv_unit(iteration, subject_groups):
    iter_paths = prepare_iteration_dirs(iteration)

    au_features = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
    gaze_raw = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

    target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

    nd_summaries, cd_ed_summaries, md_summaries = [], [], []

    # 1. ND 데이터 로드
    if os.path.exists(Config.ND_FOLDER_PATH):
        files = sorted([f for f in os.listdir(Config.ND_FOLDER_PATH) if f.endswith('.csv')])
        for filename in files:
            match = re.search(r'(T\d{3})', filename)
            if match and match.group(1) in subject_groups['ND']:
                try:
                    df = pd.read_csv(os.path.join(Config.ND_FOLDER_PATH, filename))
                    df = add_gaze_kinematic_features(df)
                    cols_to_keep = [col for col in target_features if col in df.columns]
                    summary = summarize_csv_segment(df[cols_to_keep].copy().fillna(0), target_features, filename, 'ND')
                    if summary: nd_summaries.append(summary)
                except Exception: pass

    # 2. CD_ED, MD 데이터 로드
    if os.path.exists(Config.FOLDER_PATH):
        files = sorted([f for f in os.listdir(Config.FOLDER_PATH) if f.endswith('.csv')])
        for filename in files:
            match = re.search(r'(T\d{3})', filename)
            if not match: continue
            
            subject_id = match.group(1)
            assigned_group = None
            
            if subject_id in subject_groups['CD_ED']:
                assigned_group = 'CD_ED'
            elif subject_id in subject_groups['MD']:
                assigned_group = 'MD'

            if assigned_group:
                try:
                    df = pd.read_csv(os.path.join(Config.FOLDER_PATH, filename))
                    if 'Distraction' in df.columns:
                        df = add_gaze_kinematic_features(df)
                        cols_to_keep = [col for col in target_features if col in df.columns] + ['Distraction']
                        segment = df[cols_to_keep].copy().fillna(0)
                        segment = segment[segment['Distraction'] > 0]
                        summary = summarize_csv_segment(segment, target_features, filename, assigned_group)
                        
                        if summary:
                            if assigned_group == 'CD_ED': cd_ed_summaries.append(summary)
                            elif assigned_group == 'MD': md_summaries.append(summary)
                except Exception: pass

    nd_df = pd.DataFrame(nd_summaries)
    cd_ed_df = pd.DataFrame(cd_ed_summaries)
    md_df = pd.DataFrame(md_summaries)

    # 3. 3-Class 비교 그룹 세팅
    group_sets = {
        "ND_vs_CD_ED_vs_MD": {"ND": nd_df, "CD_ED": cd_ed_df, "MD": md_df}
    }

    # 4. 정규성 검정 및 저장
    shapiro_dfs = run_shapiro_wilk(target_features, group_sets)
    
    shapiro_csv_path = iter_paths["shapiro_csv"]
    
    comp_name = "ND_vs_CD_ED_vs_MD"
    if comp_name in shapiro_dfs and not shapiro_dfs[comp_name].empty:
        shapiro_dfs[comp_name].to_csv(shapiro_csv_path, index=False, encoding='utf-8-sig')

    return shapiro_dfs

# [신규 추가] 바 그래프 시각화 함수
def plot_normality_summary(summary_df, save_dir):
    """Feature와 Group별 정규성 통과 횟수를 바 그래프로 시각화합니다."""
    # 시각화를 위해 피벗 테이블로 변환 (Index: Feature, Columns: Group)
    pivot_df = summary_df.pivot(index='Feature', columns='Group', values='Normal_Count_P>=0.05')
    
    # 보기 좋게 정렬하기 위해, 세 그룹 통과 횟수 총합을 기준으로 내림차순 정렬
    pivot_df['Total_Pass'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by='Total_Pass', ascending=False).drop(columns='Total_Pass')
    
    # 그래프 설정 및 그리기
    ax = pivot_df.plot(kind='bar', figsize=(18, 8), width=0.8, colormap='Set2')
    
    plt.title('Shapiro-Wilk Normality Test Pass Count by Feature and Group (10 Iterations)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Pass Count (Max 10)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(range(0, 11))  # y축은 0부터 10까지
    
    plt.legend(title='Group', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "Total_Normality_Summary_BarChart.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


if __name__ == "__main__":
    TOTAL_ITERATIONS = 10
    RANDOM_SEED = 42 
    random.seed(RANDOM_SEED)

    # 마스터 폴더 준비
    os.makedirs(Config.MASTER_DIR, exist_ok=True)
    
    # 가용한 모든 피험자 목록 가져오기
    all_subjects = extract_all_available_subjects()
    total_subjects = len(all_subjects)
    
    seen_assignments = set()
    all_shapiro_results = []
    
    if total_subjects < 3:
        print(f"[오류] 피험자가 너무 적습니다. 최소 3명 이상 필요합니다. (발견된 피험자: {total_subjects}명)")
    else:
        print(f"총 {total_subjects}명의 피험자를 3개 그룹(ND, CD_ED, MD)으로 분배하여 {TOTAL_ITERATIONS}회 정규성 검정(Shapiro-Wilk) 시뮬레이션을 시작합니다 (시드: {RANDOM_SEED}).\n")
        
        for i in range(1, TOTAL_ITERATIONS + 1):
            
            # 고유한 배정 조합 찾기
            while True:
                random.shuffle(all_subjects)
                
                chunk_size = total_subjects // 3
                remainder = total_subjects % 3
                sizes = [chunk_size + (1 if j < remainder else 0) for j in range(3)]
                
                subject_groups = {}
                current_idx = 0
                group_names = ['ND', 'CD_ED', 'MD']
                
                for j, group in enumerate(group_names):
                    subject_groups[group] = all_subjects[current_idx : current_idx + sizes[j]]
                    current_idx += sizes[j]
                
                assignment_signature = tuple(
                    tuple(sorted(subject_groups[g])) for g in group_names
                )
                
                if assignment_signature not in seen_assignments:
                    seen_assignments.add(assignment_signature)
                    break
            
            print(f"▶ [Iter {i}/{TOTAL_ITERATIONS}] 피험자 고유 배정 완료 및 검정 진행 중...")
            
            # 분석 수행 및 결과 누적
            iter_shapiro_dfs = test_features_normality_csv_unit(i, subject_groups)
            
            comp_name = "ND_vs_CD_ED_vs_MD"
            if comp_name in iter_shapiro_dfs and not iter_shapiro_dfs[comp_name].empty:
                all_shapiro_results.append(iter_shapiro_dfs[comp_name])
                
        # ---------------------------------------------------------
        # 10회 전체 결과 요약, 카운트 및 시각화
        # ---------------------------------------------------------
        if all_shapiro_results:
            print("\n[결과 종합] 피처 및 그룹별 정규성 만족(P>=0.05) 빈도 요약 중...")
            combined_df = pd.concat(all_shapiro_results, ignore_index=True)
            
            def summarize_normality(x):
                return pd.Series({
                    'Total_Tests': len(x),
                    'Normal_Count_P>=0.05': x['Is_Normal'].sum(),
                    'Mean_W_statistic': round(x['W_statistic'].mean(), 4),
                    'Mean_P_value': round(x['P_value'].mean(), 6)
                })

            # Feature와 Group별로 그룹화하여 요약
            summary_stats = combined_df.groupby(['Feature', 'Group']).apply(summarize_normality).reset_index()
            summary_stats = summary_stats.sort_values(by=['Normal_Count_P>=0.05', 'Feature'], ascending=[False, True])
            
            summary_path = os.path.join(Config.MASTER_DIR, "Total_Normality_Summary_3Class.csv")
            summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            print(f"\n[완료] 정규성 요약 파일 생성됨: {summary_path}")
            
            # [신규 추가] 바 그래프 시각화 실행
            print("[진행 중] 정규성 통과 빈도 바 그래프 생성 중...")
            plot_path = plot_normality_summary(summary_stats, Config.MASTER_DIR)
            print(f"[완료] 바 그래프 이미지 생성됨: {plot_path}")
            
            print("\n--- Top 5 정규성을 잘 만족하는 피처/그룹 미리보기 ---")
            print(summary_stats.head(5).to_string(index=False))
        else:
            print("[알림] 종합할 결과 데이터가 없습니다.")
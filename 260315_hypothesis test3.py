"""
===============================================================================
운전자 주의 분산(Driver Distraction) 데이터 통계 분석 (10회 반복 시뮬레이션)
===============================================================================

[주요 기능]
- 피험자 개인차(Inter-subject variability)로 인한 통계적 오류를 방지하기 위해, 
  전체 피험자 풀(Pool)에서 무작위로 ND, CD, ED, MD 그룹을 배정하는 과정을 10회 반복합니다.
- 매 반복마다 전체 피험자를 4개 그룹에 최대한 균등하게 100% 분배하여 데이터를 모두 활용합니다.
- [추가] 10회의 반복 동안 **단 한 번도 동일한 피험자 그룹 배정이 발생하지 않도록** 보장합니다.
- [추가] 10회 전체 결과를 종합하여, 피처별로 통계적 유의성(p < 0.05)이 몇 번이나 
  발생했는지 카운트한 요약 파일(Summary)을 생성합니다.
- 모든 시뮬레이션 결과(CSV, Boxplot)는 'ANOVA_Simulation_Results'라는 하나의 
  마스터 폴더 내에 저장됩니다.
===============================================================================
"""

import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg

# ==========================================
# [USER CONFIGURATION] 고정 폴더 설정
# ==========================================
class Config:
    FOLDER_PATH = "Distraction_dataset_Final_Merged_ANOVA"  # CD, ED, MD 폴더
    ND_FOLDER_PATH = "dataset_nd_merged"                    # ND 폴더
    MASTER_RESULT_DIR = "ANOVA_Simulation_Results"          # 전체 결과가 저장될 마스터 폴더
    FPS = 28
    
    # 반복문 안에서 동적으로 변경될 변수들
    WELCH_ANOVA_SAVE_PATH = ""
    GAMES_HOWELL_SAVE_PATH = ""
    ANOVA_PLOT_DIR = ""
    ANOVA_BEST_DIR = ""

def prepare_iteration_dirs(iteration):
    iter_dir = os.path.join(Config.MASTER_RESULT_DIR, f"iter_{iteration}")
    Config.ANOVA_PLOT_DIR = os.path.join(iter_dir, "welch_anova_boxplots")
    Config.ANOVA_BEST_DIR = os.path.join(Config.ANOVA_PLOT_DIR, "best")
    
    Config.WELCH_ANOVA_SAVE_PATH = os.path.join(iter_dir, f"Feature_Welch_ANOVA_Results_iter{iteration}.csv")
    Config.GAMES_HOWELL_SAVE_PATH = os.path.join(iter_dir, f"Feature_GamesHowell_Posthoc_Results_iter{iteration}.csv")

    os.makedirs(Config.ANOVA_PLOT_DIR, exist_ok=True)
    os.makedirs(Config.ANOVA_BEST_DIR, exist_ok=True)


def safe_filename(name):
    name = str(name)
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    name = name.replace("(", "").replace(")", "").replace(":", "_")
    name = name.replace("*", "_").replace("?", "_").replace('"', "_")
    name = name.replace("<", "_").replace(">", "_").replace("|", "_")
    return name


def find_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


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


def draw_and_save_multiclass_boxplot(feature, set_name, group_dict, p_val, significant):
    plot_order = ["ND", "CD", "ED", "MD"]
    available_groups = [g for g in plot_order if g in group_dict and len(group_dict[g].dropna()) > 0]
    if len(available_groups) < 2:
        return

    values_list = [pd.Series(group_dict[g]).dropna() for g in available_groups]

    filename = safe_filename(f"{feature}_{set_name}_boxplot.png")
    plot_path = os.path.join(Config.ANOVA_PLOT_DIR, filename)
    best_path = os.path.join(Config.ANOVA_BEST_DIR, filename)

    plt.figure(figsize=(9, 6))
    plt.boxplot(values_list, tick_labels=available_groups, showmeans=False)
    plt.title(f"Welch ANOVA Result - {feature}")
    plt.xticks(fontweight='bold')
    plt.ylabel("Mean Value", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_welch_anova_and_games_howell(target_features, nd_df, cd_df, ed_df, md_df):
    welch_anova_results = []
    games_howell_results = []

    group_sets = [
        ("ND_CD_ED_MD", {"ND": nd_df, "CD": cd_df, "ED": ed_df, "MD": md_df}),
    ]

    for feature in target_features:
        for set_name, group_dict in group_sets:
            merged_rows = []
            valid_group_names = []
            current_group_values = {}

            for group_name, group_df in group_dict.items():
                if group_df.empty or feature not in group_df.columns:
                    continue
                values = group_df[feature].dropna()
                if len(values) < 2:
                    continue

                valid_group_names.append(group_name)
                current_group_values[group_name] = values
                for v in values:
                    merged_rows.append({"FeatureValue": v, "Group": group_name})

            if len(valid_group_names) != len(group_dict):
                continue

            long_df = pd.DataFrame(merged_rows)
            if long_df.empty:
                continue

            try:
                welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")
                
                f_col = find_existing_column(welch_df, ["F"])
                p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
                
                f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
                p_val = welch_df.loc[0, p_col]

                is_significant = p_val < 0.05
                decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"

                row = {
                    "Feature": feature,
                    "Comparison": set_name,
                    "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
                    "P_value": p_val,
                    "Result": decision
                }

                for group_name, group_df in group_dict.items():
                    values = group_df[feature].dropna()
                    row[f"{group_name}_Mean"] = round(values.mean(), 4) if len(values) > 0 else np.nan

                welch_anova_results.append(row)
                draw_and_save_multiclass_boxplot(feature, set_name, current_group_values, p_val, is_significant)

                if is_significant:
                    gh_df = pg.pairwise_gameshowell(data=long_df, dv="FeatureValue", between="Group")
                    gh_df["Feature"] = feature
                    games_howell_results.append(gh_df)

            except Exception as e:
                pass 

    welch_anova_df = pd.DataFrame(welch_anova_results)
    games_howell_df = pd.concat(games_howell_results, ignore_index=True) if games_howell_results else pd.DataFrame()

    return welch_anova_df, games_howell_df


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


def test_features_significance_csv_unit(iteration, subject_groups):
    prepare_iteration_dirs(iteration)

    au_features = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
    gaze_raw = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

    target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

    nd_summaries, cd_summaries, ed_summaries, md_summaries = [], [], [], []

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

    # 2. CD, ED, MD 데이터 로드
    if os.path.exists(Config.FOLDER_PATH):
        files = sorted([f for f in os.listdir(Config.FOLDER_PATH) if f.endswith('.csv')])
        for filename in files:
            match = re.search(r'(T\d{3})', filename)
            if not match: continue
            
            subject_id = match.group(1)
            assigned_group = None
            for group in ['CD', 'ED', 'MD']:
                if subject_id in subject_groups[group]:
                    assigned_group = group
                    break

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
                            if assigned_group == 'CD': cd_summaries.append(summary)
                            elif assigned_group == 'ED': ed_summaries.append(summary)
                            elif assigned_group == 'MD': md_summaries.append(summary)
                except Exception: pass

    nd_df = pd.DataFrame(nd_summaries)
    cd_df = pd.DataFrame(cd_summaries)
    ed_df = pd.DataFrame(ed_summaries)
    md_df = pd.DataFrame(md_summaries)

    # 3. 통계 검정 및 저장
    welch_anova_df, games_howell_df = run_welch_anova_and_games_howell(target_features, nd_df, cd_df, ed_df, md_df)
    
    welch_anova_df.to_csv(Config.WELCH_ANOVA_SAVE_PATH, index=False, encoding='utf-8-sig')
    if not games_howell_df.empty:
        games_howell_df.to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')

    # 종합 집계를 위해 welch_anova_df를 리턴
    return welch_anova_df


if __name__ == "__main__":
    TOTAL_ITERATIONS = 10
    RANDOM_SEED = 42 
    random.seed(RANDOM_SEED)

    # 마스터 폴더 준비
    os.makedirs(Config.MASTER_RESULT_DIR, exist_ok=True)
    
    # 가용한 모든 피험자 목록 가져오기
    all_subjects = extract_all_available_subjects()
    total_subjects = len(all_subjects)
    
    # 중복 배정을 막기 위한 세트(set)
    seen_assignments = set()
    all_anova_results = []
    
    if total_subjects < 4:
        print(f"[오류] 피험자가 너무 적습니다. 최소 4명 이상 필요합니다. (발견된 피험자: {total_subjects}명)")
    else:
        print(f"총 {total_subjects}명의 피험자를 {TOTAL_ITERATIONS}회 시뮬레이션 합니다 (시드: {RANDOM_SEED}).\n")
        
        for i in range(1, TOTAL_ITERATIONS + 1):
            
            # [수정] 이전에 나온 배정 조합과 겹치지 않을 때까지 무작위 섞기 반복
            while True:
                random.shuffle(all_subjects)
                
                chunk_size = total_subjects // 4
                remainder = total_subjects % 4
                sizes = [chunk_size + (1 if j < remainder else 0) for j in range(4)]
                
                subject_groups = {}
                current_idx = 0
                group_names = ['ND', 'CD', 'ED', 'MD']
                
                for j, group in enumerate(group_names):
                    subject_groups[group] = all_subjects[current_idx : current_idx + sizes[j]]
                    current_idx += sizes[j]
                
                # 시그니처 생성 (그룹 내 순서와 무관하게 비교하기 위해 정렬)
                assignment_signature = tuple(
                    tuple(sorted(subject_groups[g])) for g in group_names
                )
                
                # 새로운 조합이면 세트에 추가하고 루프 탈출
                if assignment_signature not in seen_assignments:
                    seen_assignments.add(assignment_signature)
                    break
            
            print(f"▶ [Iter {i}/{TOTAL_ITERATIONS}] 피험자 고유 배정 완료:")
            print(f"  - ND ({len(subject_groups['ND'])}명): {subject_groups['ND']}")
            print(f"  - CD ({len(subject_groups['CD'])}명): {subject_groups['CD']}")
            print(f"  - ED ({len(subject_groups['ED'])}명): {subject_groups['ED']}")
            print(f"  - MD ({len(subject_groups['MD'])}명): {subject_groups['MD']}\n")
            
            # 분석 수행 및 결과 누적
            iter_anova_df = test_features_significance_csv_unit(i, subject_groups)
            if iter_anova_df is not None and not iter_anova_df.empty:
                all_anova_results.append(iter_anova_df)
                
        # ---------------------------------------------------------
        # 10회 전체 결과 요약 및 카운트
        # ---------------------------------------------------------
        if all_anova_results:
            print("[결과 종합] 통계적 유의성 요약 중...")
            combined_df = pd.concat(all_anova_results, ignore_index=True)
            
            # 피처(Feature)별로 묶어 유의미한(P < 0.05) 횟수 계산
            def summarize_feature(x):
                return pd.Series({
                    'Total_Iterations': len(x),
                    'Significant_Count_P<0.05': (x['P_value'] < 0.05).sum(),
                    'Mean_P_value': round(x['P_value'].mean(), 6)
                })
                
            summary_stats = combined_df.groupby('Feature').apply(summarize_feature).reset_index()
            # 유의미한 횟수가 높은 순으로 정렬
            summary_stats = summary_stats.sort_values(by='Significant_Count_P<0.05', ascending=False)
            
            summary_path = os.path.join(Config.MASTER_RESULT_DIR, "Total_Significance_Summary.csv")
            summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            print(f"\n[완료] 요약 파일이 생성되었습니다: {summary_path}")
            print("--- Top 5 유의미한 피처 미리보기 ---")
            print(summary_stats.head(5).to_string(index=False))
        else:
            print("[알림] 종합할 결과 데이터가 없습니다.")
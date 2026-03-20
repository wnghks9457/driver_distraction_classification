"""
===============================================================================
운전자 주의 분산(Driver Distraction) 데이터 통계 분석 (10회 반복 시뮬레이션)
===============================================================================

[주요 기능]
- 실험 조건: CD(인지)와 ED(감정)를 각각 분리하여 독립적인 그룹으로 분석.
- 피험자 배정: 전체 피험자 풀에서 ND, CD, ED, MD 4개 그룹으로 10회 무작위 분배.
- 비교 조건: ND vs CD vs ED vs MD (4-Class 다중 비교)
- 측정 지표: 평균(mean), 분산(var), 표준편차(std) 3가지 경우에 대해 각각 실험.
- 결과 저장: 네 그룹을 동시에 비교하는 Welch ANOVA를 수행하고, 유의미한 경우 
  Games-Howell 사후 검정을 통해 분석.
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
    FPS = 28
    
    # 4-Class 비교를 위한 단일 마스터 폴더 (실행 시 통계량에 따라 변경됨)
    MASTER_DIR_TEMPLATE = "260319_4-class_ANOVA_Simulation_Results_nd-cd-ed-md_{metric}"


def prepare_iteration_dirs(iteration, metric):
    """매 회차마다 폴더와 저장 경로를 생성하여 반환합니다."""
    master_dir = Config.MASTER_DIR_TEMPLATE.format(metric=metric)
    iter_dir = os.path.join(master_dir, f"iter_{iteration}")
    plot_dir = os.path.join(iter_dir, "welch_anova_boxplots")
    best_dir = os.path.join(plot_dir, "best")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    return {
        "plot_dir": plot_dir,
        "best_dir": best_dir,
        "anova_csv": os.path.join(iter_dir, f"Feature_Welch_ANOVA_{metric}_iter{iteration}.csv"),
        "gh_csv": os.path.join(iter_dir, f"Feature_GamesHowell_Posthoc_{metric}_iter{iteration}.csv")
    }


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


def summarize_csv_segment(df_segment, target_features, file_name, group_label, metric):
    """
    지정된 metric(평균, 분산, 표준편차)에 따라 시계열 데이터를 단일 값으로 요약합니다.
    """
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
            
        # 선택된 통계량에 따라 계산 방식 변경
        if metric == 'mean':
            summary[feature] = df_segment[feature].mean()
        elif metric == 'var':
            summary[feature] = df_segment[feature].var()
        elif metric == 'std':
            summary[feature] = df_segment[feature].std()

    return summary


def draw_and_save_multiclass_boxplot(feature, set_name, group_dict, p_val, significant, plot_dir, best_dir, metric):
    # [수정] 4-Class 구조로 Boxplot 시각화 순서 변경
    plot_order = ["ND", "CD", "ED", "MD"]
    available_groups = [g for g in plot_order if g in group_dict and len(group_dict[g].dropna()) > 0]
    if len(available_groups) < 2:
        return

    values_list = [pd.Series(group_dict[g]).dropna() for g in available_groups]

    filename = safe_filename(f"{feature}_{set_name}_{metric}_boxplot.png")
    plot_path = os.path.join(plot_dir, filename)
    best_path = os.path.join(best_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.boxplot(values_list, tick_labels=available_groups, showmeans=False)
    plt.title(f"Welch ANOVA 4-Class ({metric.upper()}) - {feature}")
    plt.xticks(fontweight='bold')
    
    ylabel_map = {'mean': "Mean Value", 'var': "Variance", 'std': "Standard Deviation"}
    plt.ylabel(ylabel_map[metric], fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_welch_anova_and_games_howell(target_features, group_sets, iter_paths, metric):
    welch_dfs = {}
    gh_dfs = {}

    for comp_name, group_dict in group_sets.items():
        welch_anova_results = []
        games_howell_results = []
        
        plot_dir = iter_paths["plot_dir"]
        best_dir = iter_paths["best_dir"]

        for feature in target_features:
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
                # 4집단 Welch ANOVA 수행
                welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")
                
                f_col = find_existing_column(welch_df, ["F"])
                p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
                
                f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
                p_val = welch_df.loc[0, p_col]

                is_significant = p_val < 0.05
                decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"

                row = {
                    "Feature": feature,
                    "Metric": metric, 
                    "Comparison": comp_name,
                    "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
                    "P_value": p_val,
                    "Result": decision
                }

                for group_name, group_df in group_dict.items():
                    values = group_df[feature].dropna()
                    row[f"{group_name}_Value"] = round(values.mean(), 4) if len(values) > 0 else np.nan

                welch_anova_results.append(row)
                draw_and_save_multiclass_boxplot(feature, comp_name, current_group_values, p_val, is_significant, plot_dir, best_dir, metric)

                if is_significant:
                    gh_df = pg.pairwise_gameshowell(data=long_df, dv="FeatureValue", between="Group")
                    gh_df["Feature"] = feature
                    gh_df["Metric"] = metric
                    games_howell_results.append(gh_df)

            except Exception as e:
                pass 

        welch_dfs[comp_name] = pd.DataFrame(welch_anova_results)
        gh_dfs[comp_name] = pd.concat(games_howell_results, ignore_index=True) if games_howell_results else pd.DataFrame()

    return welch_dfs, gh_dfs


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


def test_features_significance_csv_unit(iteration, subject_groups, metric):
    iter_paths = prepare_iteration_dirs(iteration, metric)

    au_features = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
    gaze_raw = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

    target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

    # [수정] 4개의 그룹을 위한 요약 리스트 분리
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
                    summary = summarize_csv_segment(df[cols_to_keep].copy().fillna(0), target_features, filename, 'ND', metric)
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
            
            # [수정] CD와 ED를 분리하여 할당
            if subject_id in subject_groups['CD']:
                assigned_group = 'CD'
            elif subject_id in subject_groups['ED']:
                assigned_group = 'ED'
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
                        summary = summarize_csv_segment(segment, target_features, filename, assigned_group, metric)
                        
                        if summary:
                            if assigned_group == 'CD': cd_summaries.append(summary)
                            elif assigned_group == 'ED': ed_summaries.append(summary)
                            elif assigned_group == 'MD': md_summaries.append(summary)
                except Exception: pass

    nd_df = pd.DataFrame(nd_summaries)
    cd_df = pd.DataFrame(cd_summaries)
    ed_df = pd.DataFrame(ed_summaries)
    md_df = pd.DataFrame(md_summaries)

    # 3. 4-Class 비교 그룹 세팅
    group_sets = {
        "ND_vs_CD_vs_ED_vs_MD": {"ND": nd_df, "CD": cd_df, "ED": ed_df, "MD": md_df}
    }

    # 4. 통계 검정 및 저장
    welch_dfs, gh_dfs = run_welch_anova_and_games_howell(target_features, group_sets, iter_paths, metric)
    
    anova_csv_path = iter_paths["anova_csv"]
    gh_csv_path = iter_paths["gh_csv"]
    
    comp_name = "ND_vs_CD_vs_ED_vs_MD"
    if comp_name in welch_dfs and not welch_dfs[comp_name].empty:
        welch_dfs[comp_name].to_csv(anova_csv_path, index=False, encoding='utf-8-sig')
    if comp_name in gh_dfs and not gh_dfs[comp_name].empty:
        gh_dfs[comp_name].to_csv(gh_csv_path, index=False, encoding='utf-8-sig')

    return welch_dfs


if __name__ == "__main__":
    TOTAL_ITERATIONS = 10
    RANDOM_SEED = 42 
    random.seed(RANDOM_SEED)

    all_subjects = extract_all_available_subjects()
    total_subjects = len(all_subjects)
    
    # 3가지 통계량에 대해 순차적으로 실험을 진행합니다.
    metrics_to_test = ['mean', 'var', 'std']
    
    # [수정] 4개 그룹이므로 최소 피험자를 4명으로 변경
    if total_subjects < 4:
        print(f"[오류] 피험자가 너무 적습니다. 최소 4명 이상 필요합니다. (발견된 피험자: {total_subjects}명)")
    else:
        for metric in metrics_to_test:
            print(f"\n{'='*60}")
            print(f"🚀 [실험 시작] 4-Class 통계 기준: {metric.upper()}")
            print(f"{'='*60}")
            
            master_dir = Config.MASTER_DIR_TEMPLATE.format(metric=metric)
            os.makedirs(master_dir, exist_ok=True)
            
            seen_assignments = set()
            all_anova_results = []
            
            # 매 통계 지표마다 동일한 피험자 분배를 유지하기 위해 시드 초기화
            random.seed(RANDOM_SEED)
            
            for i in range(1, TOTAL_ITERATIONS + 1):
                while True:
                    random.shuffle(all_subjects)
                    
                    # [수정] 전체 피험자를 4등분
                    chunk_size = total_subjects // 4
                    remainder = total_subjects % 4
                    sizes = [chunk_size + (1 if j < remainder else 0) for j in range(4)]
                    
                    subject_groups = {}
                    current_idx = 0
                    group_names = ['ND', 'CD', 'ED', 'MD']
                    
                    for j, group in enumerate(group_names):
                        subject_groups[group] = all_subjects[current_idx : current_idx + sizes[j]]
                        current_idx += sizes[j]
                    
                    assignment_signature = tuple(
                        tuple(sorted(subject_groups[g])) for g in group_names
                    )
                    
                    if assignment_signature not in seen_assignments:
                        seen_assignments.add(assignment_signature)
                        break
                
                print(f"▶ [{metric.upper()} - Iter {i}/{TOTAL_ITERATIONS}] 4-Class 분석 중...")
                
                # 분석 수행 및 결과 누적
                iter_anova_dfs = test_features_significance_csv_unit(i, subject_groups, metric)
                
                comp_name = "ND_vs_CD_vs_ED_vs_MD"
                if comp_name in iter_anova_dfs and not iter_anova_dfs[comp_name].empty:
                    all_anova_results.append(iter_anova_dfs[comp_name])
                    
            # 해당 metric에 대한 10회 전체 결과 요약
            if all_anova_results:
                print(f"\n[{metric.upper()} 결과 종합] 4-Class 통계적 유의성 요약 중...")
                combined_df = pd.concat(all_anova_results, ignore_index=True)
                
                def summarize_feature(x):
                    return pd.Series({
                        'Total_Iterations': len(x),
                        'Significant_Count_P<0.05': (x['P_value'] < 0.05).sum(),
                        'Mean_P_value': round(x['P_value'].mean(), 6)
                    })

                summary_stats = combined_df.groupby('Feature').apply(summarize_feature).reset_index()
                summary_stats = summary_stats.sort_values(by='Significant_Count_P<0.05', ascending=False)
                
                summary_path = os.path.join(master_dir, f"Total_Significance_Summary_{metric.upper()}_4Class.csv")
                summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
                
                print(f"[완료] 요약 파일 생성됨: {summary_path}")
                print(f"--- Top 5 유의미한 피처 ({metric.upper()}) ---")
                print(summary_stats.head(5).to_string(index=False))
            else:
                print(f"[알림] {metric.upper()}에 대한 종합할 결과 데이터가 없습니다.")
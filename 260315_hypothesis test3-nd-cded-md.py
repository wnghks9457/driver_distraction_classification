"""
===============================================================================
운전자 주의 분산(Driver Distraction) 데이터 통계 분석 (10회 반복 시뮬레이션)
===============================================================================

[주요 기능]
- 실험 조건: CD(인지)와 ED(감정)를 하나의 통합 그룹(CD_ED)으로 묶어 분석.
- 피험자 배정: 전체 피험자 풀에서 ND, CD_ED, MD 3개 그룹으로 10회 무작위 분배.
- 비교 조건: ND vs CD_ED vs MD (3-Class 다중 비교)
- 결과 저장: 세 그룹을 동시에 비교하는 Welch ANOVA를 수행하고, 유의미한 경우 
  Games-Howell 사후 검정을 통해 구체적으로 어느 그룹 간에 차이가 있는지 분석합니다.
  결과는 'ANOVA_Simulation_Results_3Class' 마스터 폴더에 종합 저장됩니다.
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
    
    # 3-Class 비교를 위한 단일 마스터 폴더
    MASTER_DIR = "ANOVA_Simulation_Results_nd-cded-md"

def prepare_iteration_dirs(iteration):
    """매 회차마다 폴더와 저장 경로를 생성하여 반환합니다."""
    iter_dir = os.path.join(Config.MASTER_DIR, f"iter_{iteration}")
    plot_dir = os.path.join(iter_dir, "welch_anova_boxplots")
    best_dir = os.path.join(plot_dir, "best")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    return {
        "plot_dir": plot_dir,
        "best_dir": best_dir,
        "anova_csv": os.path.join(iter_dir, f"Feature_Welch_ANOVA_Results_nd-cded-md_iter{iteration}.csv"),
        "gh_csv": os.path.join(iter_dir, f"Feature_GamesHowell_Posthoc_Results_nd-cded-md_iter{iteration}.csv")
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


def draw_and_save_multiclass_boxplot(feature, set_name, group_dict, p_val, significant, plot_dir, best_dir):
    plot_order = ["ND", "CD_ED", "MD"]
    available_groups = [g for g in plot_order if g in group_dict and len(group_dict[g].dropna()) > 0]
    if len(available_groups) < 2:
        return

    values_list = [pd.Series(group_dict[g]).dropna() for g in available_groups]

    filename = safe_filename(f"{feature}_{set_name}_boxplot.png")
    plot_path = os.path.join(plot_dir, filename)
    best_path = os.path.join(best_dir, filename)

    plt.figure(figsize=(9, 6))
    plt.boxplot(values_list, tick_labels=available_groups, showmeans=False)
    plt.title(f"Welch ANOVA Result (3-Class) - {feature}")
    plt.xticks(fontweight='bold')
    plt.ylabel("Mean Value", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_welch_anova_and_games_howell(target_features, group_sets, iter_paths):
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
                # 3집단(ND, CD_ED, MD) Welch ANOVA 수행
                welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")
                
                f_col = find_existing_column(welch_df, ["F"])
                p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
                
                f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
                p_val = welch_df.loc[0, p_col]

                is_significant = p_val < 0.05
                decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"

                row = {
                    "Feature": feature,
                    "Comparison": comp_name,
                    "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
                    "P_value": p_val,
                    "Result": decision
                }

                for group_name, group_df in group_dict.items():
                    values = group_df[feature].dropna()
                    row[f"{group_name}_Mean"] = round(values.mean(), 4) if len(values) > 0 else np.nan

                welch_anova_results.append(row)
                draw_and_save_multiclass_boxplot(feature, comp_name, current_group_values, p_val, is_significant, plot_dir, best_dir)

                # ANOVA 결과가 유의미하면 Games-Howell 사후 검정 수행 (3집단 간의 모든 1:1 쌍 비교)
                if is_significant:
                    gh_df = pg.pairwise_gameshowell(data=long_df, dv="FeatureValue", between="Group")
                    gh_df["Feature"] = feature
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


def test_features_significance_csv_unit(iteration, subject_groups):
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
            
            # CD와 ED를 CD_ED 그룹으로 통합
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

    # 4. 통계 검정 및 저장
    welch_dfs, gh_dfs = run_welch_anova_and_games_howell(target_features, group_sets, iter_paths)
    
    anova_csv_path = iter_paths["anova_csv"]
    gh_csv_path = iter_paths["gh_csv"]
    
    comp_name = "ND_vs_CD_ED_vs_MD"
    if comp_name in welch_dfs and not welch_dfs[comp_name].empty:
        welch_dfs[comp_name].to_csv(anova_csv_path, index=False, encoding='utf-8-sig')
    if comp_name in gh_dfs and not gh_dfs[comp_name].empty:
        gh_dfs[comp_name].to_csv(gh_csv_path, index=False, encoding='utf-8-sig')

    return welch_dfs


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
    all_anova_results = []
    
    if total_subjects < 3:
        print(f"[오류] 피험자가 너무 적습니다. 최소 3명 이상 필요합니다. (발견된 피험자: {total_subjects}명)")
    else:
        print(f"총 {total_subjects}명의 피험자를 3개 그룹(ND, CD_ED, MD)으로 분배하여 {TOTAL_ITERATIONS}회 시뮬레이션을 시작합니다 (시드: {RANDOM_SEED}).\n")
        
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
            
            print(f"▶ [Iter {i}/{TOTAL_ITERATIONS}] 피험자 고유 배정 완료:")
            print(f"  - ND ({len(subject_groups['ND'])}명): {subject_groups['ND']}")
            print(f"  - CD_ED 통합 ({len(subject_groups['CD_ED'])}명): {subject_groups['CD_ED']}")
            print(f"  - MD ({len(subject_groups['MD'])}명): {subject_groups['MD']}\n")
            
            # 분석 수행 및 결과 누적
            iter_anova_dfs = test_features_significance_csv_unit(i, subject_groups)
            
            comp_name = "ND_vs_CD_ED_vs_MD"
            if comp_name in iter_anova_dfs and not iter_anova_dfs[comp_name].empty:
                all_anova_results.append(iter_anova_dfs[comp_name])
                
        # ---------------------------------------------------------
        # 10회 전체 결과 요약 및 카운트
        # ---------------------------------------------------------
        if all_anova_results:
            print("[결과 종합] 3-Class 통계적 유의성 요약 중...")
            combined_df = pd.concat(all_anova_results, ignore_index=True)
            
            def summarize_feature(x):
                return pd.Series({
                    'Total_Iterations': len(x),
                    'Significant_Count_P<0.05': (x['P_value'] < 0.05).sum(),
                    'Mean_P_value': round(x['P_value'].mean(), 6)
                })

            summary_stats = combined_df.groupby('Feature').apply(summarize_feature).reset_index()
            summary_stats = summary_stats.sort_values(by='Significant_Count_P<0.05', ascending=False)
            
            summary_path = os.path.join(Config.MASTER_DIR, "Total_Significance_Summary_3Class.csv")
            summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            print(f"\n[완료] 요약 파일 생성됨: {summary_path}")
            print("--- Top 5 유의미한 피처 미리보기 ---")
            print(summary_stats.head(5).to_string(index=False))
        else:
            print("[알림] 종합할 결과 데이터가 없습니다.")
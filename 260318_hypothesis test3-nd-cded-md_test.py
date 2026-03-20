"""
===============================================================================
운전자 주의 분산(Driver Distraction) 데이터 통계 분석 (최적 지표 혼합 시뮬레이션)
===============================================================================

[주요 기능 및 연구 설계]
1. 실험 조건: CD(인지)와 ED(감정)를 하나의 통합 그룹(CD_ED)으로 묶어 3-Class 비교.
   (ND vs CD_ED vs MD)
2. 시뮬레이션: 피험자 개인차 통제를 위해 전체 피험자를 3집단으로 10회 무작위 분배.
   (10회 모두 단 한 번도 겹치지 않는 고유한 조합 보장)
3. [핵심] 혼합 집계(Optimal Aggregation): 39개 피처의 물리적/심리적 특성에 맞춰,
   - 얼굴 표정(AU), 시선 운동량(Kinematic), 브레이크: 상태의 '강도(Mean)' 적용
   - 차량 거동(Speed/Steering), 머리 자세(Pose), 시선 원천(Gaze Raw): 조작의 '불안정성/분산도(STD)' 적용
4. 결과 종합: 10회 반복 후 각 피처가 몇 번이나 유의미(p < 0.05)했는지 카운트.
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
    
    # 혼합 지표 적용 3-Class 비교 마스터 폴더
    MASTER_DIR = "260318_ANOVA_Simulation_Results_nd-cded-md_test"

# ==========================================
# [핵심] 39개 피처별 최적 분석 지표 (Mean vs STD) 매핑
# ==========================================
OPTIMAL_CONFIG = {
    # 1. 얼굴 표정 (Facial AUs) - 상태의 지속적 강도(Mean)가 중요
    **{f'AU{i:02d}_r': 'mean' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]},
    
    # 2. 머리 자세 (Head Pose) - 두리번거림 등 불안정성(STD)이 중요
    'pose_Tx': 'std', 'pose_Ty': 'std', 'pose_Tz': 'std',
    'pose_Rx': 'std', 'pose_Ry': 'std', 'pose_Rz': 'std',
    
    # 3. 차량 제어 (Vehicle) - 핸들 조작 변동성(SDLP) 등 제어 불안정성(STD)이 중요. 단, Brake는 빈도/강도(Mean).
    'Steering': 'std', 'LaneOffset': 'std', 'Speed': 'std', 'Acceleration': 'std',
    'Brake': 'mean',
    
    # 4. 시선 원천 데이터 (Gaze Raw) - 시선이 흩어지는 분산도(STD)가 중요
    'gaze_0_x': 'std', 'gaze_0_y': 'std', 'gaze_0_z': 'std',
    'gaze_1_x': 'std', 'gaze_1_y': 'std', 'gaze_1_z': 'std',
    'gaze_angle_x': 'std', 'gaze_angle_y': 'std',
    
    # 5. 시선 운동 피처 (Gaze Kinematic) - 눈동자가 움직이는 평균적인 속도와 폭(Mean)이 중요
    'gaze_vel': 'mean', 'gaze_amp': 'mean', 'gaze_acc': 'mean'
}

def prepare_iteration_dirs(iteration):
    iter_dir = os.path.join(Config.MASTER_DIR, f"iter_{iteration}")
    plot_dir = os.path.join(iter_dir, "welch_anova_boxplots")
    best_dir = os.path.join(plot_dir, "best")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    return {
        "plot_dir": plot_dir,
        "best_dir": best_dir,
        "anova_csv": os.path.join(iter_dir, f"Feature_Welch_ANOVA_Optimal_iter{iteration}.csv"),
        "gh_csv": os.path.join(iter_dir, f"Feature_GamesHowell_Posthoc_Optimal_iter{iteration}.csv")
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

# ==========================================
# [수정됨] 파일(세션) 단위 요약 함수
# ==========================================
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
        
        # 설정된 최적 지표(mean 또는 std)에 따라 집계
        agg_func = OPTIMAL_CONFIG.get(feature, 'mean')
        
        if agg_func == 'mean':
            summary[feature] = df_segment[feature].mean()
        elif agg_func == 'std':
            summary[feature] = df_segment[feature].std()

    return summary

def draw_and_save_multiclass_boxplot(feature, set_name, group_dict, p_val, significant, plot_dir, best_dir):
    plot_order = ["ND", "CD_ED", "MD"]
    available_groups = [g for g in plot_order if g in group_dict and len(group_dict[g].dropna()) > 0]
    if len(available_groups) < 2:
        return

    values_list = [pd.Series(group_dict[g]).dropna() for g in available_groups]

    # 그래프 제목에 어떤 지표(Mean/STD)를 썼는지 표시
    agg_method = OPTIMAL_CONFIG.get(feature, 'mean').upper()

    filename = safe_filename(f"{feature}_{set_name}_boxplot.png")
    plot_path = os.path.join(plot_dir, filename)
    best_path = os.path.join(best_dir, filename)

    plt.figure(figsize=(9, 6))
    plt.boxplot(values_list, tick_labels=available_groups, showmeans=False)
    plt.title(f"Welch ANOVA Result (3-Class) - {feature}\nAggregation: {agg_method} | p-value: {p_val:.5f}")
    plt.xticks(fontweight='bold')
    plt.ylabel(f"Aggregated Value ({agg_method})", fontweight='bold')
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
                # 3집단 Welch ANOVA 수행
                welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")
                
                f_col = find_existing_column(welch_df, ["F"])
                p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
                
                f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
                p_val = welch_df.loc[0, p_col]

                is_significant = p_val < 0.05
                decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
                agg_method = OPTIMAL_CONFIG.get(feature, 'mean').upper()

                row = {
                    "Feature": feature,
                    "Comparison": comp_name,
                    "Aggregation": agg_method,
                    "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
                    "P_value": p_val,
                    "Result": decision
                }

                for group_name, group_df in group_dict.items():
                    values = group_df[feature].dropna()
                    row[f"{group_name}_Value"] = round(values.mean(), 4) if len(values) > 0 else np.nan

                welch_anova_results.append(row)
                draw_and_save_multiclass_boxplot(feature, comp_name, current_group_values, p_val, is_significant, plot_dir, best_dir)

                # 사후 검정 (Games-Howell)
                if is_significant:
                    gh_df = pg.pairwise_gameshowell(data=long_df, dv="FeatureValue", between="Group")
                    gh_df["Feature"] = feature
                    gh_df["Aggregation"] = agg_method
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

    # 39개 분석 대상 피처 
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
    
    all_subjects = extract_all_available_subjects()
    total_subjects = len(all_subjects)
    
    seen_assignments = set()
    all_anova_results = []
    
    if total_subjects < 3:
        print(f"[오류] 피험자가 너무 적습니다. 최소 3명 이상 필요합니다. (발견된 피험자: {total_subjects}명)")
    else:
        print(f"총 {total_subjects}명의 피험자를 3개 그룹(ND, CD_ED, MD)으로 분배하여 {TOTAL_ITERATIONS}회 시뮬레이션을 시작합니다.")
        print(f"(적용된 방법: 혼합 최적화 지표 (Mean & STD) | 시드: {RANDOM_SEED})\n")
        
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
            
            print(f"▶ [Iter {i}/{TOTAL_ITERATIONS}] 피험자 배정 완료:")
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
            print("[결과 종합] 통계적 유의성 요약 중...")
            combined_df = pd.concat(all_anova_results, ignore_index=True)
            
            def summarize_feature(x):
                return pd.Series({
                    'Aggregation_Method': x['Aggregation'].iloc[0],
                    'Total_Iterations': len(x),
                    'Significant_Count_P<0.05': (x['P_value'] < 0.05).sum(),
                    'Mean_P_value': round(x['P_value'].mean(), 6)
                })

            summary_stats = combined_df.groupby('Feature').apply(summarize_feature).reset_index()
            summary_stats = summary_stats.sort_values(by=['Significant_Count_P<0.05', 'Mean_P_value'], ascending=[False, True])
            
            summary_path = os.path.join(Config.MASTER_DIR, "Total_Significance_Summary_Optimal.csv")
            summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            print(f"\n[완료] 요약 파일 생성됨: {summary_path}")
            print("--- Top 5 유의미한 피처 미리보기 ---")
            print(summary_stats.head(5).to_string(index=False))
        else:
            print("[알림] 종합할 결과 데이터가 없습니다.")
import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pingouin as pg

# ==========================================
# [USER CONFIGURATION] 설정 파라미터
# ==========================================
class Config:
    FOLDER_PATH = "Distraction_dataset_Final_Merged_ANOVA"  # CD, ED, MD 폴더
    ND_FOLDER_PATH = "dataset_nd_ANOVA"                     # ND 폴더 (얼굴/시선 및 차량 데이터 혼재)

    SAVE_PATH = "Feature_Statistical_Test_Results_CSV_Unit3.csv"
    WELCH_ANOVA_SAVE_PATH = "Feature_Welch_ANOVA_Results_CSV_Unit3.csv"
    GAMES_HOWELL_SAVE_PATH = "Feature_GamesHowell_Posthoc_Results_CSV_Unit3.csv"

    FPS = 28
    PLOT_DIR = "boxplots3"
    BEST_DIR = os.path.join(PLOT_DIR, "best")
    ANOVA_PLOT_DIR = "welch_anova_boxplots3"
    ANOVA_BEST_DIR = os.path.join(ANOVA_PLOT_DIR, "best")


def prepare_dirs():
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    os.makedirs(Config.BEST_DIR, exist_ok=True)
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


# ==========================================
# 평균만 적용하는 요약 함수
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

        # 모든 feature에 대해 평균만 사용
        summary[feature] = df_segment[feature].mean()

    return summary


def draw_and_save_boxplot(feature, comparison_name, nd_values, d_values, p_val, significant):
    nd_values = pd.Series(nd_values).dropna()
    d_values = pd.Series(d_values).dropna()
    if len(nd_values) == 0 or len(d_values) == 0:
        return

    filename = safe_filename(f"{feature}_ND_vs_{comparison_name}.png")
    plot_path = os.path.join(Config.PLOT_DIR, filename)
    best_path = os.path.join(Config.BEST_DIR, filename)

    plt.figure(figsize=(8, 6))
    plt.boxplot([nd_values, d_values], tick_labels=["ND", comparison_name], showmeans=False)
    plt.title(f"{feature} | ND vs {comparison_name}\np-value = {p_val:.6g}")
    plt.ylabel(feature)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        ("ND_CD_ED", {"ND": nd_df, "CD": cd_df, "ED": ed_df}),
        ("ND_CD_MD", {"ND": nd_df, "CD": cd_df, "MD": md_df}),
        ("ND_ED_MD", {"ND": nd_df, "ED": ed_df, "MD": md_df}),
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
                ddof1_col = find_existing_column(welch_df, ["ddof1"])
                ddof2_col = find_existing_column(welch_df, ["ddof2"])
                np2_col = find_existing_column(welch_df, ["np2", "n2"])

                f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
                p_val = welch_df.loc[0, p_col]
                ddof1 = welch_df.loc[0, ddof1_col] if ddof1_col is not None else np.nan
                ddof2 = welch_df.loc[0, ddof2_col] if ddof2_col is not None else np.nan
                np2 = welch_df.loc[0, np2_col] if np2_col is not None else np.nan

                is_significant = p_val < 0.05
                decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
                null_hypo = f"{', '.join(valid_group_names)} 그룹에서 CSV 단위 평균 ({feature})은 모두 같다."

                row = {
                    "Feature": feature,
                    "Comparison": set_name,
                    "Null_Hypothesis": null_hypo,
                    "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
                    "ddof1": round(ddof1, 4) if pd.notna(ddof1) else np.nan,
                    "ddof2": round(ddof2, 4) if pd.notna(ddof2) else np.nan,
                    "P_value": p_val,
                    "np2": round(np2, 6) if pd.notna(np2) else np.nan,
                    "Result": decision
                }

                for group_name, group_df in group_dict.items():
                    values = group_df[feature].dropna()
                    row[f"{group_name}_N_CSV"] = len(values)
                    row[f"{group_name}_Mean"] = round(values.mean(), 4) if len(values) > 0 else np.nan

                welch_anova_results.append(row)

                if set_name == "ND_CD_ED_MD":
                    draw_and_save_multiclass_boxplot(feature, set_name, current_group_values, p_val, is_significant)

                if is_significant:
                    gh_df = pg.pairwise_gameshowell(data=long_df, dv="FeatureValue", between="Group")
                    gh_df["Feature"] = feature
                    gh_df["Comparison_Set"] = set_name

                    preferred_cols = [
                        "Feature", "Comparison_Set", "A", "B",
                        "mean(A)", "mean(B)", "diff", "se", "T", "df", "pval", "hedges"
                    ]
                    existing_cols = [c for c in preferred_cols if c in gh_df.columns]
                    other_cols = [c for c in gh_df.columns if c not in existing_cols]
                    gh_df = gh_df[existing_cols + other_cols]
                    games_howell_results.append(gh_df)

            except Exception as e:
                print(f"[WELCH ANOVA ERROR] {feature} | {set_name} : {e}")

    welch_anova_df = pd.DataFrame(welch_anova_results)
    games_howell_df = pd.concat(games_howell_results, ignore_index=True) if games_howell_results else pd.DataFrame()

    return welch_anova_df, games_howell_df


def test_features_significance_csv_unit():
    prepare_dirs()

    au_features = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
        'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
        'AU25_r', 'AU26_r', 'AU45_r'
    ]
    pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
    gaze_raw = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']
    kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

    target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

    SUBJECT_GROUPS = {
        'ND': ['T002', 'T006', 'T009', 'T014', 'T017', 'T024', 'T026', 'T027', 'T031', 'T034', 'T043', 'T050'],
        'CD': ['T005', 'T010', 'T011', 'T012', 'T015', 'T021', 'T023', 'T029', 'T039', 'T045', 'T051'],
        'ED': ['T008', 'T016', 'T019', 'T022', 'T033', 'T036', 'T038', 'T040', 'T044', 'T054'],
        'MD': ['T001', 'T003', 'T004', 'T013', 'T020', 'T025', 'T032', 'T035', 'T041', 'T047']
    }

    nd_summaries, cd_summaries, ed_summaries, md_summaries = [], [], [], []

    # ------------------------------------------------------------------
    # 1. ND 폴더에서 데이터 로드 (얼굴 .avi1 파일과 차량 파일 병합)
    # ------------------------------------------------------------------
    if os.path.exists(Config.ND_FOLDER_PATH):
        all_nd_files = os.listdir(Config.ND_FOLDER_PATH)
        avi1_files = sorted([f for f in all_nd_files if '.avi1' in f and f.endswith('.csv')])

        for filename in avi1_files:
            match = re.search(r'(T\d{3})', filename)
            if not match:
                continue
            subject_id = match.group(1)

            if subject_id not in SUBJECT_GROUPS['ND']:
                continue

            file_path = os.path.join(Config.ND_FOLDER_PATH, filename)
            try:
                df = pd.read_csv(file_path)
                df = add_gaze_kinematic_features(df)
                cols_to_keep = [col for col in target_features if col in df.columns]
                df_filtered = df[cols_to_keep].copy().fillna(0)

                nd_summary = summarize_csv_segment(df_filtered, target_features, filename, 'ND')

                prefix_match = re.search(r'(T\d{3}-\d{3})', filename)
                if nd_summary and prefix_match:
                    prefix = prefix_match.group(1)
                    vehicle_filename = f"{prefix}.csv"
                    vehicle_path = os.path.join(Config.ND_FOLDER_PATH, vehicle_filename)

                    if os.path.exists(vehicle_path):
                        v_df = pd.read_csv(vehicle_path)
                        v_cols = [col for col in vehicle_features if col in v_df.columns]
                        if v_cols:
                            v_filtered = v_df[v_cols].copy().fillna(0)
                            v_summary = summarize_csv_segment(v_filtered, vehicle_features, vehicle_filename, 'ND')

                            for vf in vehicle_features:
                                if vf in v_summary and pd.notna(v_summary[vf]):
                                    nd_summary[vf] = v_summary[vf]
                    else:
                        print(f"[WARNING] {filename} 에 대응하는 차량 데이터({vehicle_filename})를 찾을 수 없습니다.")

                if nd_summary:
                    nd_summaries.append(nd_summary)

            except Exception as e:
                print(f"[ERROR] ND 파일 처리 실패 - {filename}: {e}")

    # ------------------------------------------------------------------
    # 2. CD, ED, MD 폴더에서 파일 읽기
    # ------------------------------------------------------------------
    if os.path.exists(Config.FOLDER_PATH):
        dist_files = sorted([f for f in os.listdir(Config.FOLDER_PATH) if f.endswith('.csv')])
        for filename in dist_files:
            match = re.search(r'(T\d{3})', filename)
            if not match:
                continue
            subject_id = match.group(1)

            assigned_group = None
            for group in ['CD', 'ED', 'MD']:
                if subject_id in SUBJECT_GROUPS[group]:
                    assigned_group = group
                    break

            if not assigned_group:
                continue

            file_path = os.path.join(Config.FOLDER_PATH, filename)
            try:
                df = pd.read_csv(file_path)
                if 'Distraction' not in df.columns:
                    continue

                df = add_gaze_kinematic_features(df)
                cols_to_keep = [col for col in target_features if col in df.columns] + ['Distraction']
                df_filtered = df[cols_to_keep].copy().fillna(0)

                segment = df_filtered[df_filtered['Distraction'] > 0]
                summary = summarize_csv_segment(segment, target_features, filename, assigned_group)

                if summary:
                    if assigned_group == 'CD':
                        cd_summaries.append(summary)
                    elif assigned_group == 'ED':
                        ed_summaries.append(summary)
                    elif assigned_group == 'MD':
                        md_summaries.append(summary)

            except Exception as e:
                print(f"[ERROR] Distraction 파일 처리 실패 - {filename}: {e}")

    nd_df = pd.DataFrame(nd_summaries)
    cd_df = pd.DataFrame(cd_summaries)
    ed_df = pd.DataFrame(ed_summaries)
    md_df = pd.DataFrame(md_summaries)

    dd_list = []
    if not cd_df.empty:
        dd_list.append(cd_df)
    if not ed_df.empty:
        dd_list.append(ed_df)
    if not md_df.empty:
        dd_list.append(md_df)

    dd_df = pd.concat(dd_list, ignore_index=True) if dd_list else pd.DataFrame()

    distraction_dict = {'CD': cd_df, 'ED': ed_df, 'MD': md_df, 'DD(Total)': dd_df}
    results = []

    for feature in target_features:
        if nd_df.empty or feature not in nd_df.columns:
            continue
        n_values = nd_df[feature].dropna()

        for dist_name, d_df in distraction_dict.items():
            if d_df.empty or feature not in d_df.columns:
                continue
            d_values = d_df[feature].dropna()
            if len(n_values) < 2 or len(d_values) < 2:
                continue

            t_stat, p_val = stats.ttest_ind(n_values, d_values, equal_var=False)
            is_significant = p_val < 0.05

            draw_and_save_boxplot(feature, dist_name, n_values, d_values, p_val, is_significant)

            decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
            results.append({
                'Feature': feature,
                'Comparison': f"ND vs {dist_name}",
                'Null_Hypothesis': f"ND와 {dist_name}에서 ({feature})는 차이가 없다.",
                'ND_N_CSV': len(n_values),
                'Distracted_N_CSV': len(d_values),
                'ND_Mean': round(n_values.mean(), 4),
                'Distracted_Mean': round(d_values.mean(), 4),
                'T_statistic': round(t_stat, 4),
                'P_value': p_val,
                'Result': decision
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(Config.SAVE_PATH, index=False, encoding='utf-8-sig')

    welch_anova_df, games_howell_df = run_welch_anova_and_games_howell(
        target_features, nd_df, cd_df, ed_df, md_df
    )
    welch_anova_df.to_csv(Config.WELCH_ANOVA_SAVE_PATH, index=False, encoding='utf-8-sig')

    if not games_howell_df.empty:
        games_howell_df.to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame().to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')

    return results_df, welch_anova_df, games_howell_df


if __name__ == "__main__":
    stat_results, welch_anova_results, games_howell_results = test_features_significance_csv_unit()

    if stat_results is not None and not stat_results.empty:
        print("\n[T-TEST 결과 미리보기]")
        print(stat_results.head())

    if welch_anova_results is not None and not welch_anova_results.empty:
        print("\n[Welch ANOVA 결과 미리보기]")
        print(welch_anova_results.head())

    if games_howell_results is not None and not games_howell_results.empty:
        print("\n[Games-Howell 결과 미리보기]")
        print(games_howell_results.head())
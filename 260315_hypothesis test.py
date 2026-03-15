# import os
# import re
# import pandas as pd
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import pingouin as pg

# # ==========================================
# # [USER CONFIGURATION] 설정 파라미터
# # ==========================================
# class Config:
#     FOLDER_PATH = "Distraction_dataset_Final_Merged"
#     SAVE_PATH = "Feature_Statistical_Test_Results_CSV_Unit.csv"
#     WELCH_ANOVA_SAVE_PATH = "Feature_Welch_ANOVA_Results_CSV_Unit.csv"
#     GAMES_HOWELL_SAVE_PATH = "Feature_GamesHowell_Posthoc_Results_CSV_Unit.csv"
#     FPS = 28

#     PLOT_DIR = "boxplots"
#     BEST_DIR = os.path.join(PLOT_DIR, "best")


# # ==========================================
# # 폴더 생성
# # ==========================================
# def prepare_dirs():
#     os.makedirs(Config.PLOT_DIR, exist_ok=True)
#     os.makedirs(Config.BEST_DIR, exist_ok=True)


# # ==========================================
# # 파일명 안전하게 변경
# # ==========================================
# def safe_filename(name):
#     name = str(name)
#     name = name.replace(" ", "_")
#     name = name.replace("/", "_")
#     name = name.replace("\\", "_")
#     name = name.replace("(", "")
#     name = name.replace(")", "")
#     name = name.replace(":", "_")
#     name = name.replace("*", "_")
#     name = name.replace("?", "_")
#     name = name.replace('"', "_")
#     name = name.replace("<", "_")
#     name = name.replace(">", "_")
#     name = name.replace("|", "_")
#     return name


# # ==========================================
# # gaze 파생 feature 계산
# # ==========================================
# def add_gaze_kinematic_features(df):
#     df = df.copy()

#     if 'timestamp' in df.columns:
#         dt = df['timestamp'].diff().replace(0, np.nan)
#         mean_dt = dt.mean() if not np.isnan(dt.mean()) else (1 / Config.FPS)
#         dt = dt.fillna(mean_dt)
#     else:
#         dt = pd.Series([1 / Config.FPS] * len(df), index=df.index)

#     if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
#         dx = df['gaze_angle_x'].diff()
#         dy = df['gaze_angle_y'].diff()

#         amp_rad = np.sqrt(dx**2 + dy**2).fillna(0)
#         df['gaze_amp'] = np.degrees(amp_rad)

#         df['gaze_vel'] = (df['gaze_amp'] / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

#         d_vel = df['gaze_vel'].diff()
#         df['gaze_acc'] = (d_vel / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

#         if len(df) > 0:
#             df.loc[df.index[0], 'gaze_acc'] = 0
#         if len(df) > 1:
#             df.loc[df.index[1], 'gaze_acc'] = 0
#     else:
#         df['gaze_amp'] = 0.0
#         df['gaze_vel'] = 0.0
#         df['gaze_acc'] = 0.0

#     return df


# # ==========================================
# # CSV 구간 평균 요약
# # ==========================================
# def summarize_csv_segment(df_segment, target_features, file_name, group_label):
#     if df_segment.empty:
#         return None

#     summary = {
#         'Source_File': file_name,
#         'Group': group_label,
#         'Num_Frames': len(df_segment)
#     }

#     for feature in target_features:
#         summary[feature] = df_segment[feature].mean() if feature in df_segment.columns else np.nan

#     return summary


# # ==========================================
# # box plot 그리기
# # ==========================================
# def draw_and_save_boxplot(feature, comparison_name, nd_values, d_values, p_val, significant):
#     nd_values = pd.Series(nd_values).dropna()
#     d_values = pd.Series(d_values).dropna()

#     if len(nd_values) == 0 or len(d_values) == 0:
#         return

#     filename = safe_filename(f"{feature}_ND_vs_{comparison_name}.png")
#     plot_path = os.path.join(Config.PLOT_DIR, filename)
#     best_path = os.path.join(Config.BEST_DIR, filename)

#     plt.figure(figsize=(8, 6))
#     plt.boxplot(
#         [nd_values, d_values],
#         tick_labels=["ND", comparison_name],
#         showmeans=True
#     )
#     plt.title(f"{feature} | ND vs {comparison_name}\np-value = {p_val:.6g}")
#     plt.ylabel(feature)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     if significant:
#         plt.savefig(best_path, dpi=300, bbox_inches='tight')

#     plt.close()


# # ==========================================
# # Welch ANOVA + Games-Howell
# # ==========================================
# def run_welch_anova_and_games_howell(target_features, nd_df, cd_df, ed_df, md_df):
#     welch_anova_results = []
#     games_howell_results = []

#     group_sets = [
#         ("ND_CD_ED", {"ND": nd_df, "CD": cd_df, "ED": ed_df}),
#         ("ND_CD_MD", {"ND": nd_df, "CD": cd_df, "MD": md_df}),
#         ("ND_ED_MD", {"ND": nd_df, "ED": ed_df, "MD": md_df}),
#         ("ND_CD_ED_MD", {"ND": nd_df, "CD": cd_df, "ED": ed_df, "MD": md_df}),
#     ]

#     for feature in target_features:
#         for set_name, group_dict in group_sets:
#             merged_rows = []
#             valid_group_names = []

#             for group_name, group_df in group_dict.items():
#                 if group_df.empty or feature not in group_df.columns:
#                     continue

#                 values = group_df[feature].dropna()
#                 if len(values) < 2:
#                     continue

#                 valid_group_names.append(group_name)

#                 for v in values:
#                     merged_rows.append({
#                         "FeatureValue": v,
#                         "Group": group_name
#                     })

#             # Welch ANOVA는 최소 2개 이상 그룹이 필요하지만,
#             # 사용자가 요청한 조합 의미를 살리려면 해당 조합의 모든 그룹이 살아있는 것이 좋음
#             if len(valid_group_names) != len(group_dict):
#                 print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : 일부 그룹 데이터 부족")
#                 continue

#             long_df = pd.DataFrame(merged_rows)

#             if long_df.empty:
#                 print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : long_df 비어 있음")
#                 continue

#             try:
#                 welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")

#                 # pingouin 결과 컬럼 예시:
#                 # Source, ddof1, ddof2, F, p_unc, np2
#                 f_stat = welch_df.loc[0, "F"]
#                 p_val = welch_df.loc[0, "p_unc"]
#                 ddof1 = welch_df.loc[0, "ddof1"]
#                 ddof2 = welch_df.loc[0, "ddof2"]
#                 np2 = welch_df.loc[0, "np2"] if "np2" in welch_df.columns else np.nan

#                 is_significant = p_val < 0.05
#                 decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
#                 null_hypo = f"{', '.join(valid_group_names)} 그룹에서 CSV 단위 평균 ({feature})은 모두 같다."

#                 row = {
#                     "Feature": feature,
#                     "Comparison": set_name,
#                     "Null_Hypothesis": null_hypo,
#                     "Welch_F": round(f_stat, 4),
#                     "ddof1": round(ddof1, 4),
#                     "ddof2": round(ddof2, 4),
#                     "P_value": p_val,
#                     "np2": round(np2, 6) if pd.notna(np2) else np.nan,
#                     "Result": decision
#                 }

#                 for group_name, group_df in group_dict.items():
#                     values = group_df[feature].dropna()
#                     row[f"{group_name}_N_CSV"] = len(values)
#                     row[f"{group_name}_Mean"] = round(values.mean(), 4)

#                 welch_anova_results.append(row)

#                 # -----------------------------
#                 # Games-Howell post-hoc
#                 # -----------------------------
#                 if is_significant:
#                     gh_df = pg.pairwise_gameshowell(
#                         data=long_df,
#                         dv="FeatureValue",
#                         between="Group"
#                     )

#                     # 주요 컬럼 예시:
#                     # A, B, mean(A), mean(B), diff, se, T, df, pval, hedges
#                     gh_df["Feature"] = feature
#                     gh_df["Comparison_Set"] = set_name

#                     # 보기 좋게 컬럼 순서 조정
#                     preferred_cols = [
#                         "Feature",
#                         "Comparison_Set",
#                         "A",
#                         "B",
#                         "mean(A)",
#                         "mean(B)",
#                         "diff",
#                         "se",
#                         "T",
#                         "df",
#                         "pval",
#                         "hedges"
#                     ]
#                     existing_cols = [c for c in preferred_cols if c in gh_df.columns]
#                     other_cols = [c for c in gh_df.columns if c not in existing_cols]
#                     gh_df = gh_df[existing_cols + other_cols]

#                     games_howell_results.append(gh_df)

#             except Exception as e:
#                 print(f"[WELCH ANOVA ERROR] {feature} | {set_name} : {e}")

#     welch_anova_df = pd.DataFrame(welch_anova_results)

#     if games_howell_results:
#         games_howell_df = pd.concat(games_howell_results, ignore_index=True)
#     else:
#         games_howell_df = pd.DataFrame()

#     return welch_anova_df, games_howell_df


# # ==========================================
# # 메인 함수
# # ==========================================
# def test_features_significance_csv_unit():
#     prepare_dirs()

#     au_features = [
#         'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
#         'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
#         'AU25_r', 'AU26_r', 'AU45_r'
#     ]
#     pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
#     vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
#     gaze_raw = [
#         'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
#         'gaze_angle_x', 'gaze_angle_y'
#     ]
#     kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

#     target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

#     csv_files = sorted([
#         os.path.join(Config.FOLDER_PATH, f)
#         for f in os.listdir(Config.FOLDER_PATH)
#         if f.endswith('.csv')
#     ])

#     nd_summaries = []
#     cd_summaries = []
#     ed_summaries = []
#     md_summaries = []

#     for file_path in csv_files:
#         filename = os.path.basename(file_path)

#         match = re.search(r'T\d+-(\d+)', filename)
#         if not match:
#             print(f"[SKIP] 파일명 규칙 불일치: {filename}")
#             continue

#         task_code = match.group(1)
#         if task_code == '005':
#             dist_type = 'CD'
#         elif task_code == '006':
#             dist_type = 'ED'
#         elif task_code == '007':
#             dist_type = 'MD'
#         else:
#             print(f"[SKIP] 분석 대상 task code 아님: {filename}")
#             continue

#         try:
#             df = pd.read_csv(file_path)

#             if 'Distraction' not in df.columns:
#                 print(f"[SKIP] Distraction 컬럼 없음: {filename}")
#                 continue

#             df = add_gaze_kinematic_features(df)

#             cols_to_keep = [col for col in target_features if col in df.columns] + ['Distraction']
#             df_filtered = df[cols_to_keep].copy().fillna(0)

#             nd_segment = df_filtered[df_filtered['Distraction'] == 0]
#             dd_segment = df_filtered[df_filtered['Distraction'] > 0]

#             nd_summary = summarize_csv_segment(nd_segment, target_features, filename, 'ND')
#             dd_summary = summarize_csv_segment(dd_segment, target_features, filename, dist_type)

#             if nd_summary is not None:
#                 nd_summaries.append(nd_summary)

#             if dd_summary is not None:
#                 if dist_type == 'CD':
#                     cd_summaries.append(dd_summary)
#                 elif dist_type == 'ED':
#                     ed_summaries.append(dd_summary)
#                 elif dist_type == 'MD':
#                     md_summaries.append(dd_summary)

#         except Exception as e:
#             print(f"[ERROR] {filename}: {e}")

#     nd_df = pd.DataFrame(nd_summaries)
#     cd_df = pd.DataFrame(cd_summaries)
#     ed_df = pd.DataFrame(ed_summaries)
#     md_df = pd.DataFrame(md_summaries)

#     dd_list = []
#     if not cd_df.empty:
#         dd_list.append(cd_df)
#     if not ed_df.empty:
#         dd_list.append(ed_df)
#     if not md_df.empty:
#         dd_list.append(md_df)

#     dd_df = pd.concat(dd_list, ignore_index=True) if dd_list else pd.DataFrame()

#     distraction_dict = {
#         'CD': cd_df,
#         'ED': ed_df,
#         'MD': md_df,
#         'DD(Total)': dd_df
#     }

#     results = []

#     # --------------------------------------
#     # 1) Welch's t-test
#     # --------------------------------------
#     for feature in target_features:
#         if nd_df.empty or feature not in nd_df.columns:
#             continue

#         n_values = nd_df[feature].dropna()

#         for dist_name, d_df in distraction_dict.items():
#             if d_df.empty or feature not in d_df.columns:
#                 continue

#             d_values = d_df[feature].dropna()

#             if len(n_values) < 2 or len(d_values) < 2:
#                 print(f"[SKIP] {feature} | ND vs {dist_name} : 표본 수 부족 (ND={len(n_values)}, D={len(d_values)})")
#                 continue

#             t_stat, p_val = stats.ttest_ind(n_values, d_values, equal_var=False)
#             is_significant = p_val < 0.05

#             draw_and_save_boxplot(
#                 feature=feature,
#                 comparison_name=dist_name,
#                 nd_values=n_values,
#                 d_values=d_values,
#                 p_val=p_val,
#                 significant=is_significant
#             )

#             decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
#             null_hypo = f"정상 주행(ND)과 주의산만 주행({dist_name})에서 CSV 단위 평균 ({feature})는 차이가 없다."

#             results.append({
#                 'Feature': feature,
#                 'Comparison': f"ND vs {dist_name}",
#                 'Null_Hypothesis': null_hypo,
#                 'ND_N_CSV': len(n_values),
#                 'Distracted_N_CSV': len(d_values),
#                 'ND_Mean': round(n_values.mean(), 4),
#                 'Distracted_Mean': round(d_values.mean(), 4),
#                 'T_statistic': round(t_stat, 4),
#                 'P_value': p_val,
#                 'Result': decision
#             })

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(Config.SAVE_PATH, index=False, encoding='utf-8-sig')

#     # --------------------------------------
#     # 2) Welch ANOVA + Games-Howell
#     # --------------------------------------
#     welch_anova_df, games_howell_df = run_welch_anova_and_games_howell(
#         target_features=target_features,
#         nd_df=nd_df,
#         cd_df=cd_df,
#         ed_df=ed_df,
#         md_df=md_df
#     )

#     welch_anova_df.to_csv(Config.WELCH_ANOVA_SAVE_PATH, index=False, encoding='utf-8-sig')

#     if not games_howell_df.empty:
#         games_howell_df.to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')
#     else:
#         pd.DataFrame().to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')

#     print(f"\nWelch's t-test 완료. 결과가 '{Config.SAVE_PATH}'에 저장되었습니다.")
#     print(f"Welch ANOVA 완료. 결과가 '{Config.WELCH_ANOVA_SAVE_PATH}'에 저장되었습니다.")
#     print(f"Games-Howell 완료. 결과가 '{Config.GAMES_HOWELL_SAVE_PATH}'에 저장되었습니다.")
#     print(f"전체 box plot 저장 폴더: {Config.PLOT_DIR}")
#     print(f"유의한 결과(best) 저장 폴더: {Config.BEST_DIR}")

#     return results_df, welch_anova_df, games_howell_df


# if __name__ == "__main__":
#     stat_results, welch_anova_results, games_howell_results = test_features_significance_csv_unit()

#     if stat_results is not None and not stat_results.empty:
#         print("\n[T-TEST 결과 미리보기]")
#         print(stat_results.head())
#     else:
#         print("생성된 t-test 결과가 없습니다.")

#     if welch_anova_results is not None and not welch_anova_results.empty:
#         print("\n[Welch ANOVA 결과 미리보기]")
#         print(welch_anova_results.head())
#     else:
#         print("생성된 Welch ANOVA 결과가 없습니다.")

#     if games_howell_results is not None and not games_howell_results.empty:
#         print("\n[Games-Howell 결과 미리보기]")
#         print(games_howell_results.head())
#     else:
#         print("생성된 Games-Howell 결과가 없습니다.")


#############################################################################################################################

# import os
# import re
# import pandas as pd
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import pingouin as pg


# # ==========================================
# # [USER CONFIGURATION] 설정 파라미터
# # ==========================================
# class Config:
#     FOLDER_PATH = "Distraction_dataset_Final_Merged"

#     SAVE_PATH = "Feature_Statistical_Test_Results_CSV_Unit.csv"
#     WELCH_ANOVA_SAVE_PATH = "Feature_Welch_ANOVA_Results_CSV_Unit.csv"
#     GAMES_HOWELL_SAVE_PATH = "Feature_GamesHowell_Posthoc_Results_CSV_Unit.csv"

#     FPS = 28

#     # ND vs D boxplot
#     PLOT_DIR = "boxplots"
#     BEST_DIR = os.path.join(PLOT_DIR, "best")

#     # Welch ANOVA multiclass boxplot
#     ANOVA_PLOT_DIR = "welch_anova_boxplots"
#     ANOVA_BEST_DIR = os.path.join(ANOVA_PLOT_DIR, "best")


# # ==========================================
# # 폴더 생성
# # ==========================================
# def prepare_dirs():
#     os.makedirs(Config.PLOT_DIR, exist_ok=True)
#     os.makedirs(Config.BEST_DIR, exist_ok=True)
#     os.makedirs(Config.ANOVA_PLOT_DIR, exist_ok=True)
#     os.makedirs(Config.ANOVA_BEST_DIR, exist_ok=True)


# # ==========================================
# # 파일명 안전하게 변경
# # ==========================================
# def safe_filename(name):
#     name = str(name)
#     name = name.replace(" ", "_")
#     name = name.replace("/", "_")
#     name = name.replace("\\", "_")
#     name = name.replace("(", "")
#     name = name.replace(")", "")
#     name = name.replace(":", "_")
#     name = name.replace("*", "_")
#     name = name.replace("?", "_")
#     name = name.replace('"', "_")
#     name = name.replace("<", "_")
#     name = name.replace(">", "_")
#     name = name.replace("|", "_")
#     return name


# # ==========================================
# # 컬럼명 자동 탐색
# # ==========================================
# def find_existing_column(df, candidates):
#     for col in candidates:
#         if col in df.columns:
#             return col
#     return None


# # ==========================================
# # gaze 파생 feature 계산
# # ==========================================
# def add_gaze_kinematic_features(df):
#     df = df.copy()

#     if 'timestamp' in df.columns:
#         dt = df['timestamp'].diff().replace(0, np.nan)
#         mean_dt = dt.mean() if not np.isnan(dt.mean()) else (1 / Config.FPS)
#         dt = dt.fillna(mean_dt)
#     else:
#         dt = pd.Series([1 / Config.FPS] * len(df), index=df.index)

#     if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
#         dx = df['gaze_angle_x'].diff()
#         dy = df['gaze_angle_y'].diff()

#         amp_rad = np.sqrt(dx**2 + dy**2).fillna(0)
#         df['gaze_amp'] = np.degrees(amp_rad)

#         df['gaze_vel'] = (df['gaze_amp'] / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

#         d_vel = df['gaze_vel'].diff()
#         df['gaze_acc'] = (d_vel / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

#         if len(df) > 0:
#             df.loc[df.index[0], 'gaze_acc'] = 0
#         if len(df) > 1:
#             df.loc[df.index[1], 'gaze_acc'] = 0
#     else:
#         df['gaze_amp'] = 0.0
#         df['gaze_vel'] = 0.0
#         df['gaze_acc'] = 0.0

#     return df


# # ==========================================
# # CSV 구간 평균 요약
# # ==========================================
# def summarize_csv_segment(df_segment, target_features, file_name, group_label):
#     if df_segment.empty:
#         return None

#     summary = {
#         'Source_File': file_name,
#         'Group': group_label,
#         'Num_Frames': len(df_segment)
#     }

#     for feature in target_features:
#         summary[feature] = df_segment[feature].mean() if feature in df_segment.columns else np.nan

#     return summary


# # ==========================================
# # ND vs D box plot 그리기
# # ==========================================
# def draw_and_save_boxplot(feature, comparison_name, nd_values, d_values, p_val, significant):
#     nd_values = pd.Series(nd_values).dropna()
#     d_values = pd.Series(d_values).dropna()

#     if len(nd_values) == 0 or len(d_values) == 0:
#         return

#     filename = safe_filename(f"{feature}_ND_vs_{comparison_name}.png")
#     plot_path = os.path.join(Config.PLOT_DIR, filename)
#     best_path = os.path.join(Config.BEST_DIR, filename)

#     plt.figure(figsize=(8, 6))
#     plt.boxplot(
#         [nd_values, d_values],
#         tick_labels=["ND", comparison_name],
#         showmeans=False
#     )
#     plt.title(f"{feature} | ND vs {comparison_name}\np-value = {p_val:.6g}")
#     plt.ylabel(feature)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     if significant:
#         plt.savefig(best_path, dpi=300, bbox_inches='tight')

#     plt.close()


# # ==========================================
# # Welch ANOVA용 multiclass box plot
# # ==========================================
# def draw_and_save_multiclass_boxplot(feature, set_name, group_dict, p_val, significant):
#     plot_order = ["ND", "CD", "ED", "MD"]
#     available_groups = [g for g in plot_order if g in group_dict and len(group_dict[g].dropna()) > 0]

#     if len(available_groups) < 2:
#         return

#     values_list = [pd.Series(group_dict[g]).dropna() for g in available_groups]

#     filename = safe_filename(f"{feature}_{set_name}_boxplot.png")
#     plot_path = os.path.join(Config.ANOVA_PLOT_DIR, filename)
#     best_path = os.path.join(Config.ANOVA_BEST_DIR, filename)

#     plt.figure(figsize=(9, 6))
#     plt.boxplot(
#         values_list,
#         tick_labels=available_groups,
#         showmeans=False
#     )
#     plt.title(f"{feature} | {set_name}\nWelch ANOVA p-value = {p_val:.6g}")
#     plt.ylabel(feature)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     if significant:
#         plt.savefig(best_path, dpi=300, bbox_inches='tight')

#     plt.close()


# # ==========================================
# # Welch ANOVA + Games-Howell
# # ==========================================
# def run_welch_anova_and_games_howell(target_features, nd_df, cd_df, ed_df, md_df):
#     welch_anova_results = []
#     games_howell_results = []

#     group_sets = [
#         ("ND_CD_ED", {"ND": nd_df, "CD": cd_df, "ED": ed_df}),
#         ("ND_CD_MD", {"ND": nd_df, "CD": cd_df, "MD": md_df}),
#         ("ND_ED_MD", {"ND": nd_df, "ED": ed_df, "MD": md_df}),
#         ("ND_CD_ED_MD", {"ND": nd_df, "CD": cd_df, "ED": ed_df, "MD": md_df}),
#     ]

#     for feature in target_features:
#         for set_name, group_dict in group_sets:
#             merged_rows = []
#             valid_group_names = []

#             current_group_values = {}

#             for group_name, group_df in group_dict.items():
#                 if group_df.empty or feature not in group_df.columns:
#                     continue

#                 values = group_df[feature].dropna()

#                 if len(values) < 2:
#                     continue

#                 valid_group_names.append(group_name)
#                 current_group_values[group_name] = values

#                 for v in values:
#                     merged_rows.append({
#                         "FeatureValue": v,
#                         "Group": group_name
#                     })

#             if len(valid_group_names) != len(group_dict):
#                 print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : 일부 그룹 데이터 부족")
#                 continue

#             long_df = pd.DataFrame(merged_rows)

#             if long_df.empty:
#                 print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : long_df 비어 있음")
#                 continue

#             try:
#                 welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")

#                 # pingouin 버전별 컬럼명 대응
#                 f_col = find_existing_column(welch_df, ["F"])
#                 p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
#                 ddof1_col = find_existing_column(welch_df, ["ddof1"])
#                 ddof2_col = find_existing_column(welch_df, ["ddof2"])
#                 np2_col = find_existing_column(welch_df, ["np2"])

#                 if p_col is None:
#                     raise ValueError(f"Welch ANOVA 결과 파일의 p-value 컬럼을 찾지 못했습니다. 현재 컬럼: {welch_df.columns.tolist()}")

#                 f_stat = welch_df.loc[0, f_col] if f_col is not None else np.nan
#                 p_val = welch_df.loc[0, p_col]
#                 ddof1 = welch_df.loc[0, ddof1_col] if ddof1_col is not None else np.nan
#                 ddof2 = welch_df.loc[0, ddof2_col] if ddof2_col is not None else np.nan
#                 np2 = welch_df.loc[0, np2_col] if np2_col is not None else np.nan

#                 is_significant = p_val < 0.05
#                 decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
#                 null_hypo = f"{', '.join(valid_group_names)} 그룹에서 CSV 단위 평균 ({feature})은 모두 같다."

#                 row = {
#                     "Feature": feature,
#                     "Comparison": set_name,
#                     "Null_Hypothesis": null_hypo,
#                     "Welch_F": round(f_stat, 4) if pd.notna(f_stat) else np.nan,
#                     "ddof1": round(ddof1, 4) if pd.notna(ddof1) else np.nan,
#                     "ddof2": round(ddof2, 4) if pd.notna(ddof2) else np.nan,
#                     "P_value": p_val,
#                     "np2": round(np2, 6) if pd.notna(np2) else np.nan,
#                     "Result": decision
#                 }

#                 for group_name, group_df in group_dict.items():
#                     values = group_df[feature].dropna()
#                     row[f"{group_name}_N_CSV"] = len(values)
#                     row[f"{group_name}_Mean"] = round(values.mean(), 4) if len(values) > 0 else np.nan

#                 welch_anova_results.append(row)

#                 # 사용자가 원하는 ND/CD/ED/MD 전체 box plot
#                 if set_name == "ND_CD_ED_MD":
#                     draw_and_save_multiclass_boxplot(
#                         feature=feature,
#                         set_name=set_name,
#                         group_dict=current_group_values,
#                         p_val=p_val,
#                         significant=is_significant
#                     )

#                 # -----------------------------
#                 # Games-Howell post-hoc
#                 # -----------------------------
#                 if is_significant:
#                     gh_df = pg.pairwise_gameshowell(
#                         data=long_df,
#                         dv="FeatureValue",
#                         between="Group"
#                     )

#                     gh_df["Feature"] = feature
#                     gh_df["Comparison_Set"] = set_name

#                     preferred_cols = [
#                         "Feature",
#                         "Comparison_Set",
#                         "A",
#                         "B",
#                         "mean(A)",
#                         "mean(B)",
#                         "diff",
#                         "se",
#                         "T",
#                         "df",
#                         "pval",
#                         "hedges"
#                     ]
#                     existing_cols = [c for c in preferred_cols if c in gh_df.columns]
#                     other_cols = [c for c in gh_df.columns if c not in existing_cols]
#                     gh_df = gh_df[existing_cols + other_cols]

#                     games_howell_results.append(gh_df)

#             except Exception as e:
#                 print(f"[WELCH ANOVA ERROR] {feature} | {set_name} : {e}")

#     welch_anova_df = pd.DataFrame(welch_anova_results)

#     if games_howell_results:
#         games_howell_df = pd.concat(games_howell_results, ignore_index=True)
#     else:
#         games_howell_df = pd.DataFrame()

#     return welch_anova_df, games_howell_df


# # ==========================================
# # 메인 함수
# # ==========================================
# def test_features_significance_csv_unit():
#     prepare_dirs()

#     au_features = [
#         'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
#         'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
#         'AU25_r', 'AU26_r', 'AU45_r'
#     ]
#     pose_features = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
#     vehicle_features = ['Speed', 'Acceleration', 'Brake', 'Steering', 'LaneOffset']
#     gaze_raw = [
#         'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
#         'gaze_angle_x', 'gaze_angle_y'
#     ]
#     kinematic_features = ['gaze_vel', 'gaze_amp', 'gaze_acc']

#     target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

#     csv_files = sorted([
#         os.path.join(Config.FOLDER_PATH, f)
#         for f in os.listdir(Config.FOLDER_PATH)
#         if f.endswith('.csv')
#     ])

#     nd_summaries = []
#     cd_summaries = []
#     ed_summaries = []
#     md_summaries = []

#     for file_path in csv_files:
#         filename = os.path.basename(file_path)

#         match = re.search(r'T\d+-(\d+)', filename)
#         if not match:
#             print(f"[SKIP] 파일명 규칙 불일치: {filename}")
#             continue

#         task_code = match.group(1)
#         if task_code == '005':
#             dist_type = 'CD'
#         elif task_code == '006':
#             dist_type = 'ED'
#         elif task_code == '007':
#             dist_type = 'MD'
#         else:
#             print(f"[SKIP] 분석 대상 task code 아님: {filename}")
#             continue

#         try:
#             df = pd.read_csv(file_path)

#             if 'Distraction' not in df.columns:
#                 print(f"[SKIP] Distraction 컬럼 없음: {filename}")
#                 continue

#             df = add_gaze_kinematic_features(df)

#             cols_to_keep = [col for col in target_features if col in df.columns] + ['Distraction']
#             df_filtered = df[cols_to_keep].copy().fillna(0)

#             nd_segment = df_filtered[df_filtered['Distraction'] == 0]
#             dd_segment = df_filtered[df_filtered['Distraction'] > 0]

#             nd_summary = summarize_csv_segment(nd_segment, target_features, filename, 'ND')
#             dd_summary = summarize_csv_segment(dd_segment, target_features, filename, dist_type)

#             if nd_summary is not None:
#                 nd_summaries.append(nd_summary)

#             if dd_summary is not None:
#                 if dist_type == 'CD':
#                     cd_summaries.append(dd_summary)
#                 elif dist_type == 'ED':
#                     ed_summaries.append(dd_summary)
#                 elif dist_type == 'MD':
#                     md_summaries.append(dd_summary)

#         except Exception as e:
#             print(f"[ERROR] {filename}: {e}")

#     nd_df = pd.DataFrame(nd_summaries)
#     cd_df = pd.DataFrame(cd_summaries)
#     ed_df = pd.DataFrame(ed_summaries)
#     md_df = pd.DataFrame(md_summaries)

#     dd_list = []
#     if not cd_df.empty:
#         dd_list.append(cd_df)
#     if not ed_df.empty:
#         dd_list.append(ed_df)
#     if not md_df.empty:
#         dd_list.append(md_df)

#     dd_df = pd.concat(dd_list, ignore_index=True) if dd_list else pd.DataFrame()

#     distraction_dict = {
#         'CD': cd_df,
#         'ED': ed_df,
#         'MD': md_df,
#         'DD(Total)': dd_df
#     }

#     results = []

#     # --------------------------------------
#     # 1) Welch's t-test
#     # --------------------------------------
#     for feature in target_features:
#         if nd_df.empty or feature not in nd_df.columns:
#             continue

#         n_values = nd_df[feature].dropna()

#         for dist_name, d_df in distraction_dict.items():
#             if d_df.empty or feature not in d_df.columns:
#                 continue

#             d_values = d_df[feature].dropna()

#             if len(n_values) < 2 or len(d_values) < 2:
#                 print(f"[SKIP] {feature} | ND vs {dist_name} : 표본 수 부족 (ND={len(n_values)}, D={len(d_values)})")
#                 continue

#             t_stat, p_val = stats.ttest_ind(n_values, d_values, equal_var=False)
#             is_significant = p_val < 0.05

#             draw_and_save_boxplot(
#                 feature=feature,
#                 comparison_name=dist_name,
#                 nd_values=n_values,
#                 d_values=d_values,
#                 p_val=p_val,
#                 significant=is_significant
#             )

#             decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
#             null_hypo = f"정상 주행(ND)과 주의산만 주행({dist_name})에서 CSV 단위 평균 ({feature})는 차이가 없다."

#             results.append({
#                 'Feature': feature,
#                 'Comparison': f"ND vs {dist_name}",
#                 'Null_Hypothesis': null_hypo,
#                 'ND_N_CSV': len(n_values),
#                 'Distracted_N_CSV': len(d_values),
#                 'ND_Mean': round(n_values.mean(), 4),
#                 'Distracted_Mean': round(d_values.mean(), 4),
#                 'T_statistic': round(t_stat, 4),
#                 'P_value': p_val,
#                 'Result': decision
#             })

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(Config.SAVE_PATH, index=False, encoding='utf-8-sig')

#     # --------------------------------------
#     # 2) Welch ANOVA + Games-Howell
#     # --------------------------------------
#     welch_anova_df, games_howell_df = run_welch_anova_and_games_howell(
#         target_features=target_features,
#         nd_df=nd_df,
#         cd_df=cd_df,
#         ed_df=ed_df,
#         md_df=md_df
#     )

#     welch_anova_df.to_csv(Config.WELCH_ANOVA_SAVE_PATH, index=False, encoding='utf-8-sig')

#     if not games_howell_df.empty:
#         games_howell_df.to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')
#     else:
#         pd.DataFrame().to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')

#     print(f"\nWelch's t-test 완료. 결과가 '{Config.SAVE_PATH}'에 저장되었습니다.")
#     print(f"Welch ANOVA 완료. 결과가 '{Config.WELCH_ANOVA_SAVE_PATH}'에 저장되었습니다.")
#     print(f"Games-Howell 완료. 결과가 '{Config.GAMES_HOWELL_SAVE_PATH}'에 저장되었습니다.")
#     print(f"ND vs D box plot 저장 폴더: {Config.PLOT_DIR}")
#     print(f"ND vs D 유의 결과(best) 저장 폴더: {Config.BEST_DIR}")
#     print(f"Welch ANOVA multiclass box plot 저장 폴더: {Config.ANOVA_PLOT_DIR}")
#     print(f"Welch ANOVA 유의 결과(best) 저장 폴더: {Config.ANOVA_BEST_DIR}")

#     return results_df, welch_anova_df, games_howell_df


# if __name__ == "__main__":
#     stat_results, welch_anova_results, games_howell_results = test_features_significance_csv_unit()

#     if stat_results is not None and not stat_results.empty:
#         print("\n[T-TEST 결과 미리보기]")
#         print(stat_results.head())
#     else:
#         print("생성된 t-test 결과가 없습니다.")

#     if welch_anova_results is not None and not welch_anova_results.empty:
#         print("\n[Welch ANOVA 결과 미리보기]")
#         print(welch_anova_results.head())
#     else:
#         print("생성된 Welch ANOVA 결과가 없습니다.")

#     if games_howell_results is not None and not games_howell_results.empty:
#         print("\n[Games-Howell 결과 미리보기]")
#         print(games_howell_results.head())
#     else:
#         print("생성된 Games-Howell 결과가 없습니다.")

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
    FOLDER_PATH = "Distraction_dataset_Final_Merged"

    SAVE_PATH = "Feature_Statistical_Test_Results_CSV_Unit2.csv"
    WELCH_ANOVA_SAVE_PATH = "Feature_Welch_ANOVA_Results_CSV_Unit2.csv"
    GAMES_HOWELL_SAVE_PATH = "Feature_GamesHowell_Posthoc_Results_CSV_Unit2.csv"

    FPS = 28

    # ND vs D boxplot
    PLOT_DIR = "boxplots2"
    BEST_DIR = os.path.join(PLOT_DIR, "best")

    # Welch ANOVA multiclass boxplot
    ANOVA_PLOT_DIR = "welch_anova_boxplots2"
    ANOVA_BEST_DIR = os.path.join(ANOVA_PLOT_DIR, "best")


# ==========================================
# 폴더 생성
# ==========================================
def prepare_dirs():
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    os.makedirs(Config.BEST_DIR, exist_ok=True)
    os.makedirs(Config.ANOVA_PLOT_DIR, exist_ok=True)
    os.makedirs(Config.ANOVA_BEST_DIR, exist_ok=True)


# ==========================================
# 파일명 안전하게 변경
# ==========================================
def safe_filename(name):
    name = str(name)
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(":", "_")
    name = name.replace("*", "_")
    name = name.replace("?", "_")
    name = name.replace('"', "_")
    name = name.replace("<", "_")
    name = name.replace(">", "_")
    name = name.replace("|", "_")
    return name


# ==========================================
# 컬럼명 자동 탐색
# ==========================================
def find_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ==========================================
# gaze 파생 feature 계산
# ==========================================
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
# subject-wise normalization (CSV 단위 z-score)
# ==========================================
def subject_zscore_normalization(df, target_features):
    df = df.copy()

    for feature in target_features:
        if feature not in df.columns:
            continue

        mean_val = df[feature].mean()
        std_val = df[feature].std()

        if pd.isna(std_val) or std_val == 0:
            continue

        df[feature] = (df[feature] - mean_val) / std_val

    return df


# ==========================================
# CSV 구간 평균 요약
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
        summary[feature] = df_segment[feature].mean() if feature in df_segment.columns else np.nan

    return summary


# ==========================================
# ND vs D box plot 그리기
# ==========================================
def draw_and_save_boxplot(feature, comparison_name, nd_values, d_values, p_val, significant):
    nd_values = pd.Series(nd_values).dropna()
    d_values = pd.Series(d_values).dropna()

    if len(nd_values) == 0 or len(d_values) == 0:
        return

    filename = safe_filename(f"{feature}_ND_vs_{comparison_name}.png")
    plot_path = os.path.join(Config.PLOT_DIR, filename)
    best_path = os.path.join(Config.BEST_DIR, filename)

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [nd_values, d_values],
        tick_labels=["ND", comparison_name],
        showmeans=False
    )
    plt.title(f"{feature} | ND vs {comparison_name}\np-value = {p_val:.6g}")
    plt.ylabel(feature)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')

    plt.close()


# ==========================================
# Welch ANOVA용 multiclass box plot
# ==========================================
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
    plt.boxplot(
        values_list,
        tick_labels=available_groups,
        showmeans=False
    )
    plt.title(f"Welch ANOVA Result - {feature}")
    plt.xticks(fontweight='bold')
    plt.ylabel(f"z-score", fontweight='bold') 

    # Y축 범위 고정 로직 추가
    if feature == "gaze_acc":
        plt.ylim(-0.001, 0.001)
    else:
        plt.ylim(-1.05, 1.05)   
        
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    if significant:
        plt.savefig(best_path, dpi=300, bbox_inches='tight')

    plt.close()


# ==========================================
# Welch ANOVA + Games-Howell
# ==========================================
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
                    merged_rows.append({
                        "FeatureValue": v,
                        "Group": group_name
                    })

            if len(valid_group_names) != len(group_dict):
                print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : 일부 그룹 데이터 부족")
                continue

            long_df = pd.DataFrame(merged_rows)

            if long_df.empty:
                print(f"[WELCH ANOVA SKIP] {feature} | {set_name} : long_df 비어 있음")
                continue

            try:
                welch_df = pg.welch_anova(data=long_df, dv="FeatureValue", between="Group")

                f_col = find_existing_column(welch_df, ["F"])
                p_col = find_existing_column(welch_df, ["p_unc", "p-unc", "P_value", "pval", "p_value"])
                ddof1_col = find_existing_column(welch_df, ["ddof1"])
                ddof2_col = find_existing_column(welch_df, ["ddof2"])
                np2_col = find_existing_column(welch_df, ["np2", "n2"])

                if p_col is None:
                    raise ValueError(
                        f"Welch ANOVA 결과의 p-value 컬럼을 찾지 못했습니다. 현재 컬럼: {welch_df.columns.tolist()}"
                    )

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
                    draw_and_save_multiclass_boxplot(
                        feature=feature,
                        set_name=set_name,
                        group_dict=current_group_values,
                        p_val=p_val,
                        significant=is_significant
                    )

                if is_significant:
                    gh_df = pg.pairwise_gameshowell(
                        data=long_df,
                        dv="FeatureValue",
                        between="Group"
                    )

                    gh_df["Feature"] = feature
                    gh_df["Comparison_Set"] = set_name

                    preferred_cols = [
                        "Feature",
                        "Comparison_Set",
                        "A",
                        "B",
                        "mean(A)",
                        "mean(B)",
                        "diff",
                        "se",
                        "T",
                        "df",
                        "pval",
                        "hedges"
                    ]
                    existing_cols = [c for c in preferred_cols if c in gh_df.columns]
                    other_cols = [c for c in gh_df.columns if c not in existing_cols]
                    gh_df = gh_df[existing_cols + other_cols]

                    games_howell_results.append(gh_df)

            except Exception as e:
                print(f"[WELCH ANOVA ERROR] {feature} | {set_name} : {e}")

    welch_anova_df = pd.DataFrame(welch_anova_results)

    if games_howell_results:
        games_howell_df = pd.concat(games_howell_results, ignore_index=True)
    else:
        games_howell_df = pd.DataFrame()

    return welch_anova_df, games_howell_df


# ==========================================
# 메인 함수
# ==========================================
def test_features_significance_csv_unit():
    prepare_dirs()

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

    target_features = au_features + pose_features + vehicle_features + gaze_raw + kinematic_features

    csv_files = sorted([
        os.path.join(Config.FOLDER_PATH, f)
        for f in os.listdir(Config.FOLDER_PATH)
        if f.endswith('.csv')
    ])

    nd_summaries = []
    cd_summaries = []
    ed_summaries = []
    md_summaries = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)

        match = re.search(r'T\d+-(\d+)', filename)
        if not match:
            print(f"[SKIP] 파일명 규칙 불일치: {filename}")
            continue

        task_code = match.group(1)
        if task_code == '005':
            dist_type = 'CD'
        elif task_code == '006':
            dist_type = 'ED'
        elif task_code == '007':
            dist_type = 'MD'
        else:
            print(f"[SKIP] 분석 대상 task code 아님: {filename}")
            continue

        try:
            df = pd.read_csv(file_path)

            if 'Distraction' not in df.columns:
                print(f"[SKIP] Distraction 컬럼 없음: {filename}")
                continue

            df = add_gaze_kinematic_features(df)

            cols_to_keep = [col for col in target_features if col in df.columns] + ['Distraction']
            df_filtered = df[cols_to_keep].copy().fillna(0)

            # --------------------------------------
            # subject-wise normalization 추가
            # --------------------------------------
            df_filtered = subject_zscore_normalization(df_filtered, target_features)

            nd_segment = df_filtered[df_filtered['Distraction'] == 0]
            dd_segment = df_filtered[df_filtered['Distraction'] > 0]

            nd_summary = summarize_csv_segment(nd_segment, target_features, filename, 'ND')
            dd_summary = summarize_csv_segment(dd_segment, target_features, filename, dist_type)

            if nd_summary is not None:
                nd_summaries.append(nd_summary)

            if dd_summary is not None:
                if dist_type == 'CD':
                    cd_summaries.append(dd_summary)
                elif dist_type == 'ED':
                    ed_summaries.append(dd_summary)
                elif dist_type == 'MD':
                    md_summaries.append(dd_summary)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

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

    distraction_dict = {
        'CD': cd_df,
        'ED': ed_df,
        'MD': md_df,
        'DD(Total)': dd_df
    }

    results = []

    # --------------------------------------
    # 1) Welch's t-test
    # --------------------------------------
    for feature in target_features:
        if nd_df.empty or feature not in nd_df.columns:
            continue

        n_values = nd_df[feature].dropna()

        for dist_name, d_df in distraction_dict.items():
            if d_df.empty or feature not in d_df.columns:
                continue

            d_values = d_df[feature].dropna()

            if len(n_values) < 2 or len(d_values) < 2:
                print(f"[SKIP] {feature} | ND vs {dist_name} : 표본 수 부족 (ND={len(n_values)}, D={len(d_values)})")
                continue

            t_stat, p_val = stats.ttest_ind(n_values, d_values, equal_var=False)
            is_significant = p_val < 0.05

            draw_and_save_boxplot(
                feature=feature,
                comparison_name=dist_name,
                nd_values=n_values,
                d_values=d_values,
                p_val=p_val,
                significant=is_significant
            )

            decision = "기각 (차이 있음)" if is_significant else "기각하지 못함 (차이 없음)"
            null_hypo = f"정상 주행(ND)과 주의산만 주행({dist_name})에서 CSV 단위 평균 ({feature})는 차이가 없다."

            results.append({
                'Feature': feature,
                'Comparison': f"ND vs {dist_name}",
                'Null_Hypothesis': null_hypo,
                'ND_N_CSV': len(n_values),
                'Distracted_N_CSV': len(d_values),
                'ND_Mean': round(n_values.mean(), 4),
                'Distracted_Mean': round(d_values.mean(), 4),
                'T_statistic': round(t_stat, 4),
                'P_value': p_val,
                'Result': decision
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(Config.SAVE_PATH, index=False, encoding='utf-8-sig')

    # --------------------------------------
    # 2) Welch ANOVA + Games-Howell
    # --------------------------------------
    welch_anova_df, games_howell_df = run_welch_anova_and_games_howell(
        target_features=target_features,
        nd_df=nd_df,
        cd_df=cd_df,
        ed_df=ed_df,
        md_df=md_df
    )

    welch_anova_df.to_csv(Config.WELCH_ANOVA_SAVE_PATH, index=False, encoding='utf-8-sig')

    if not games_howell_df.empty:
        games_howell_df.to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame().to_csv(Config.GAMES_HOWELL_SAVE_PATH, index=False, encoding='utf-8-sig')

    print(f"\nWelch's t-test 완료. 결과가 '{Config.SAVE_PATH}'에 저장되었습니다.")
    print(f"Welch ANOVA 완료. 결과가 '{Config.WELCH_ANOVA_SAVE_PATH}'에 저장되었습니다.")
    print(f"Games-Howell 완료. 결과가 '{Config.GAMES_HOWELL_SAVE_PATH}'에 저장되었습니다.")
    print(f"ND vs D box plot 저장 폴더: {Config.PLOT_DIR}")
    print(f"ND vs D 유의 결과(best) 저장 폴더: {Config.BEST_DIR}")
    print(f"Welch ANOVA multiclass box plot 저장 폴더: {Config.ANOVA_PLOT_DIR}")
    print(f"Welch ANOVA 유의 결과(best) 저장 폴더: {Config.ANOVA_BEST_DIR}")

    return results_df, welch_anova_df, games_howell_df


if __name__ == "__main__":
    stat_results, welch_anova_results, games_howell_results = test_features_significance_csv_unit()

    if stat_results is not None and not stat_results.empty:
        print("\n[T-TEST 결과 미리보기]")
        print(stat_results.head())
    else:
        print("생성된 t-test 결과가 없습니다.")

    if welch_anova_results is not None and not welch_anova_results.empty:
        print("\n[Welch ANOVA 결과 미리보기]")
        print(welch_anova_results.head())
    else:
        print("생성된 Welch ANOVA 결과가 없습니다.")

    if games_howell_results is not None and not games_howell_results.empty:
        print("\n[Games-Howell 결과 미리보기]")
        print(games_howell_results.head())
    else:
        print("생성된 Games-Howell 결과가 없습니다.")
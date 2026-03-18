import pandas as pd
import os
import glob

# 1. 폴더 경로 설정 (실제 경로로 수정하세요)
folder_a = 'dataset_nd'
folder_b = 'nd_4_extracted'
output_folder = 'dataset_nd_merged'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. A 폴더의 *.avi1.csv 파일 목록 가져오기
file_list_a = glob.glob(os.path.join(folder_a, "*.avi1.csv"))

for file_path_a in file_list_a:
    file_name_a = os.path.basename(file_path_a)
    
    # 매칭될 B 파일 이름 생성 (예: T001-004.csv)
    file_name_b = file_name_a.replace(".avi1.csv", ".csv")
    file_path_b = os.path.join(folder_b, file_name_b)
    
    if os.path.exists(file_path_b):
        print(f"작업 중: {file_name_a} & {file_name_b}")
        
        # 데이터 로드
        df_a = pd.read_csv(file_path_a)
        df_b = pd.read_csv(file_path_b)
        
        # [A 데이터 처리]
        df_a['Distraction'] = 0
        # timestamp 0.0 ~ 0.99 -> merge_key 0 으로 변환
        df_a['merge_key'] = df_a['timestamp'].astype(int)
        
        # [B 데이터 처리] 
        # Time 1 -> merge_key 0 으로 변환 (A의 0초대와 매칭하기 위함)
        df_b['merge_key'] = df_b['Time'] - 1
        
        # 3. 시간 기준 병합 (A를 기준으로 B 데이터를 반복해서 붙임)
        # B의 1초 데이터(key 0)가 A의 0.x초(key 0) 모든 행에 붙게 됨
        merged_df = pd.merge(df_a, df_b, on='merge_key', how='left')
        
        # 병합용 임시 키 삭제
        merged_df.drop(columns=['merge_key'], inplace=True)
        
        # 4. 파일명 변경 및 저장
        # 형식: T001-004.avi1_merged_labeled_processed.csv
        new_file_name = file_name_a.replace(".csv", "_merged_labeled_processed.csv")
        save_path = os.path.join(output_folder, new_file_name)
        
        merged_df.to_csv(save_path, index=False)
        print(f"완료: {new_file_name} 저장 성공")
    else:
        print(f"건너뜀: {file_name_b} 파일을 찾을 수 없습니다.")

print("-" * 30)
print("모든 파일의 통합 작업이 완료되었습니다.")
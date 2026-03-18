import os
import glob
import pandas as pd

# 1. 대상 폴더 및 결과를 저장할 새 폴더 경로 설정
folder_path = 'nd_4'
output_folder = 'nd_4_extracted'  # 원본과 구분하기 위해 결과용 폴더 지정

# 결과 저장 폴더가 없다면 새로 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 'nd_4' 폴더 안의 모든 CSV 파일 경로 가져오기
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

total_files = len(csv_files)
files_with_data = 0
total_extracted_rows = 0

print(f"작업 시작: 총 {total_files}개의 CSV 파일을 검사합니다.\n")
print("-" * 50)

# 3. 각 CSV 파일을 순회하며 데이터 추출 및 개별 저장
for file in csv_files:
    filename = os.path.basename(file)
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file)
        
        # 'Drive' 컬럼이 존재하는지 먼저 확인
        if 'Drive' in df.columns:
            # Drive 값이 4인 데이터만 필터링
            filtered_df = df[df['Drive'] == 4]
            
            # 필터링된 데이터가 비어있는지(없는 경우) 확인
            if filtered_df.empty:
                print(f"[{filename}] Drive가 4인 데이터가 없습니다. (패스)")
            else:
                # 파일명과 확장자 분리 (예: 'T001.csv' -> 'T001', '.csv')
                base_name, ext = os.path.splitext(filename)
                
                # 추출된 데이터를 개별 파일로 저장 (예: T001-004.csv)
                output_filename = f"{base_name}-004{ext}"
                output_path = os.path.join(output_folder, output_filename)
                
                filtered_df.to_csv(output_path, index=False)
                
                rows = len(filtered_df)
                total_extracted_rows += rows
                files_with_data += 1
                print(f"[{filename}] {rows}건 추출 완료 -> '{output_filename}'에 저장됨")
        else:
            print(f"[{filename}] 'Drive' 컬럼 자체가 없습니다. (패스)")
            
    except Exception as e:
        print(f"[{filename}] 파일을 읽는 중 오류 발생: {e}")

# 4. 최종 결과 요약 출력
print("-" * 50)
print("\n[데이터 추출 최종 결과 요약]")
print(f"- 검사한 총 파일 수: {total_files}개")

if files_with_data > 0:
    print(f"- 데이터를 발견하여 저장한 파일 수: {files_with_data}개")
    print(f"- 총 추출된 데이터(행) 수: {total_extracted_rows}건")
    print(f"- 저장 위치: '{output_folder}' 폴더에 개별 CSV 파일로 저장 완료")
else:
    print("- 조건(Drive=4)에 맞는 데이터가 포함된 파일이 하나도 없습니다.")
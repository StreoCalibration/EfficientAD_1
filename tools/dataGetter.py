import os
import shutil
import re

def collect_and_rename_images(base_dir, output_dir):
    """
    지정된 폴더 구조에서 이미지 파일들을 수집하고 순차적으로 이름을 변경하여 저장합니다.

    :param base_dir: 최상위 소스 경로 (예: Z:\\KohYoung\\NIS\\...)
    :param output_dir: 결과물을 저장할 경로
    """
    # 1. 결과물을 저장할 폴더가 없으면 생성합니다.
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과물 저장 폴더: '{output_dir}'")

    # 파일명에 사용할 카운터 초기화
    file_counter = 0

    try:
        # 2. base_dir 안의 모든 폴더 목록을 가져옵니다.
        # 폴더 이름 앞의 숫자를 기준으로 정렬하기 위한 함수
        def get_leading_number(folder_name):
            match = re.match(r'^(\d+)_', folder_name)
            return int(match.group(1)) if match else -1

        # '_ObjectFindingColorDifference'를 포함하고 디렉터리인 것만 필터링
        target_folders = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and '_ObjectFindingColorDifference' in d
        ]
        
        # 숫자를 기준으로 폴더 정렬
        target_folders.sort(key=get_leading_number)

    except FileNotFoundError:
        print(f"[오류] 기본 경로를 찾을 수 없습니다: '{base_dir}'")
        return

    # 3. 정렬된 각 폴더에 대해 작업 반복
    for folder_name in target_folders:
        print(f"\n처리 중인 폴더: {folder_name}")
        out_path = os.path.join(base_dir, folder_name, 'out')

        if not os.path.isdir(out_path):
            print(f"  - '{out_path}' 경로를 찾을 수 없어 건너뜁니다.")
            continue

        # 4. 고정된 하위 폴더와 이미지 파일 목록
        sub_folders = ['m_imgTopRGB', 'm_imgMidRGB', 'm_imgBotRGB']
        image_files_to_find = ['img0.png', 'img1.png', 'img2.png']

        # 하위 폴더 순회 (Top -> Mid -> Bot)
        for sub_folder in sub_folders:
            # 이미지 파일 순회 (img0 -> img1 -> img2)
            for img_file in image_files_to_find:
                source_path = os.path.join(out_path, sub_folder, 'canvases', img_file)

                # 5. 원본 파일이 존재하는지 확인
                if os.path.exists(source_path):
                    # 6. 새 파일 이름과 경로 생성
                    destination_path = os.path.join(output_dir, f"{file_counter}.png")
                    
                    # 7. 파일 복사
                    shutil.copy2(source_path, destination_path)
                    print(f"  - 복사 완료: '{source_path}' -> '{destination_path}'")
                    
                    # 8. 다음 파일 이름을 위해 카운터 증가
                    file_counter += 1
                else:
                    print(f"  - 파일 없음 (건너뛰기): '{source_path}'")

    print("\n--------------------")
    print("      ✨ 작업 완료 ✨      ")
    print(f"총 {file_counter}개의 파일을 '{output_dir}' 폴더에 성공적으로 복사했습니다.")
    print("--------------------")


# --- 실행 부분 ---
if __name__ == "__main__":
    # ❗️사용자 설정: 본인의 환경에 맞게 이 두 경로를 수정하세요.
    # r""'는 경로의 백슬래시(\)를 문자로 인식하게 해줍니다.
    
    # 1. 원본 폴더들이 있는 상위 경로
    SOURCE_BASE_DIRECTORY = r"Z:\KohYoung\NIS\Simulation\Kohyoung\NIS\DCS\VisionRecipeDump"
    
    # 2. 결과물을 저장할 폴더 경로
    DESTINATION_DIRECTORY = r"Z:\KohYoung\NIS\Simulation\Kohyoung\NIS\DCS\VisionRecipeDump\Collected_Images"

    # 함수 실행
    collect_and_rename_images(SOURCE_BASE_DIRECTORY, DESTINATION_DIRECTORY)
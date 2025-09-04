import httpx
import asyncio
import json
import polars as pl
from tqdm.asyncio import tqdm_asyncio
import os

# --- 설정 ---
# 중요: 국가법령정보 공동활용 사이트에서 발급받은 인증키를 입력하세요.
# 환경 변수에서 키를 먼저 읽어오고, 없으면 아래 문자열을 사용합니다.
API_KEY = os.environ.get("LAW_API_KEY", "YOUR_API_KEY")

# 검색할 키워드
SEARCH_QUERY = "노동"
# 결과를 저장할 파일명
OUTPUT_PARQUET_FILE = "korean_labor_law_precedents_api.parquet"
# 동시 요청 수 (API 서버 정책에 따라 조절)
MAX_CONCURRENT_REQUESTS = 5
# 한 페이지에 요청할 판례 수 (최대 100)
DISPLAY_COUNT = 100

# --- API 정보 ---
# 판례 목록 조회 API URL
PRECEDENT_LIST_URL = "https://www.law.go.kr/DRF/lawService.do"
# 판례 본문 조회 API URL
PRECEDENT_DETAIL_URL = "https://www.law.go.kr/DRF/lawService.do"

def parse_precedent_list(json_data):
    """판례 목록 JSON 응답에서 판례 일련번호 리스트를 추출합니다."""
    try:
        data = json.loads(json_data)
        if "PrecSearch" in data:
            items = data["PrecSearch"]
            serial_numbers = [item.get("판례일련번호") for item in items]
            total_count = int(data.get("totalCnt", "0"))
            return serial_numbers, total_count
        else:
            # Check for error message
            if "ERROR" in data:
                print(f"API 오류: {data['ERROR']['MESSAGE']}")
            return [], 0
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return [], 0

def parse_precedent_detail(json_data):
    """판례 본문 JSON 응답에서 상세 정보를 추출하여 딕셔너리로 반환합니다."""
    try:
        data = json.loads(json_data)
        return {
            "case_number": data.get("사건번호", "N/A"),
            "judgment_date": data.get("선고일자", "N/A"),
            "court_name": data.get("법원명", "N/A"),
            "case_name": data.get("사건명", "N/A"),
            "judgment_summary": data.get("판결요지", "N/A"),
            "referenced_statutes": data.get("참조조문", "N/A"),
        }
    except json.JSONDecodeError:
        return None

async def fetch_precedent_details(client, serial_num, pbar):
    """판례 일련번호를 사용하여 상세 정보를 비동기적으로 가져옵니다."""
    params = {
        "OC": API_KEY,
        "target": "prec",
        "ID": serial_num,
        "type": "JSON",
    }
    try:
        response = await client.get(PRECEDENT_DETAIL_URL, params=params, timeout=20)
        response.raise_for_status()
        pbar.update(1)
        return parse_precedent_detail(response.text)
    except httpx.RequestError as e:
        print(f"판례 상세 정보 요청 실패: {serial_num} - {e}")
        pbar.update(1)
        return None

async def main():
    """메인 실행 함수"""
    print("="*50)
    print("법제처 Open API 기반 판례 크롤링 스크립트 (JSON, Polars, Parquet)")
    print("="*50)

    if API_KEY == "YOUR_API_KEY":
        print("오류: API 인증키가 설정되지 않았습니다.")
        print("스크립트 상단의 'YOUR_API_KEY'를 발급받은 인증키로 변경해주세요.")
        return

    all_serial_numbers = []
    page = 1
    total_count = 0

    print(f"'{SEARCH_QUERY}' 키워드로 전체 판례 목록을 가져오는 중...")

    async with httpx.AsyncClient() as client:
        # 1. 전체 판례 일련번호 수집 (페이지네이션)
        while True:
            params = {
                "OC": API_KEY,
                "target": "prec",
                "query": SEARCH_QUERY,
                "display": DISPLAY_COUNT,
                "page": page,
                "type": "JSON",
            }
            try:
                response = await client.get(PRECEDENT_LIST_URL, params=params, timeout=20)
                response.raise_for_status()
                
                serial_numbers, current_total = parse_precedent_list(response.text)
                
                if page == 1:
                    total_count = current_total
                    if total_count == 0:
                        print("검색 결과가 없습니다. 검색어를 확인해주세요.")
                        return
                    print(f"총 {total_count}개의 판례를 발견했습니다.")

                if not serial_numbers:
                    break
                
                all_serial_numbers.extend(serial_numbers)
                print(f"페이지 {page}: {len(serial_numbers)}개 수집 (총 {len(all_serial_numbers)}/{total_count})")

                if len(all_serial_numbers) >= total_count:
                    break
                
                page += 1
                await asyncio.sleep(0.5) # API 서버 부하 감소

            except httpx.RequestError as e:
                print(f"판례 목록 요청 실패 (페이지 {page}): {e}")
                break
            except Exception as e:
                print(f"처리 중 오류 발생: {e}")
                break

        if not all_serial_numbers:
            print("수집된 판례가 없습니다. 스크립트를 종료합니다.")
            return

        print(f"
총 {len(all_serial_numbers)}개의 판례 상세 정보 수집을 시작합니다.")

        # 2. 병렬/비동기 처리를 통한 상세 정보 수집
        progress_bar = tqdm_asyncio(total=len(all_serial_numbers), desc="판례 상세 정보 수집 중")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def fetch_detail(serial_num):
            async with semaphore:
                return await fetch_precedent_details(client, serial_num, progress_bar)

        tasks = [fetch_detail(sn) for sn in all_serial_numbers]
        results = await asyncio.gather(*tasks)
        progress_bar.close()

        # 3. Parquet 파일로 저장
        filtered_results = [res for res in results if res is not None]
        if filtered_results:
            df = pl.DataFrame(filtered_results)
            df.write_parquet(OUTPUT_PARQUET_FILE)
            print(f"\n총 {len(df)}개의 판례를 '{OUTPUT_PARQUET_FILE}' 파일에 저장했습니다.")
        else:
            print("\n수집된 판례가 없어 파일을 저장하지 않았습니다.")


    print("\n" + "="*50)
    print(f"크롤링 완료! 결과가 '{OUTPUT_PARQUET_FILE}' 파일에 저장되었습니다.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
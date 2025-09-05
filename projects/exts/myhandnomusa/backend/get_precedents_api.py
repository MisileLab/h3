import httpx
import asyncio
import json
import polars as pl
import os
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

_ = load_dotenv()

# --- 설정 ---
# 중요: 국가법령정보 공동활용 사이트에서 발급받은 인증키를 입력하세요.
# 환경 변수(LAW_API_KEY)에서 키를 읽어오거나, 아래 문자열을 직접 수정하세요.
API_KEY = os.environ.get("LAW_API_KEY", "YOUR_API_KEY")

# 검색할 키워드
SEARCH_QUERY = "노동"
# 결과를 저장할 파일명
OUTPUT_PARQUET_FILE = "korean_labor_precedents.parquet"
# 한 페이지에 요청할 판례 수 (최대 100)
DISPLAY_COUNT = 100
MAX_CONCURRENT_REQUESTS = 5  # 동시 요청 수 (API 서버 정책에 따라 조절)

# --- API 정보 ---
# 판례 목록 조회 API URL
PRECEDENT_LIST_URL = "http://www.law.go.kr/DRF/lawSearch.do"

def parse_precedent_list(json_data):
    """판례 목록 JSON 응답에서 판례 리스트를 추출합니다."""
    try:
        data = json.loads(json_data)
        # API 응답 키를 유연하게 확인합니다.
        for key in ["PrecSearch", "lawSearch", "Prec", "prec"]:
            if key in data and "prec" in data[key]:
                search_data = data[key]
                items = search_data.get("prec", [])
                total_count = int(search_data.get("totalCnt", "0"))
                return items, total_count
        
        if "prec" in data:
            items = data["prec"]
            total_count = int(data.get("totalCnt", "0"))
            return items, total_count
            
        elif "ERROR" in data:
            print(f"API 오류: {data['ERROR']['MESSAGE']}")
            return [], 0
        elif "faultInfo" in data:
            print(f"API 오류: {data['faultInfo']['message']}")
            return [], 0
        else:
            # 응답에 'totalCnt'가 있는지 확인하여 목록 데이터인지 추정
            if "totalCnt" in data:
                items = data.get("prec", [])
                total_count = int(data.get("totalCnt", "0"))
                return items, total_count
            else:
                print(f"알 수 없는 오류 또는 데이터 없음: {json_data}")
                return [], 0
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}\n데이터: {json_data}")
        return [], 0

async def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("법제처 Open API 기반 판례 목록 크롤링 스크립트")
    print("=" * 50)

    if API_KEY == "YOUR_API_KEY":
        print("오류: API 키 미설정. 환경 변수 'LAW_API_KEY'를 설정하세요.")
        return

    all_precedents = []
    page = 1
    total_count = 0
    
    progress_bar = None

    async with httpx.AsyncClient() as client:
        print(f"'{SEARCH_QUERY}' 키워드로 전체 판례 목록을 가져오는 중...")
        while True:
            # target을 'prec'로 설정
            params = {"OC": API_KEY, "target": "prec", "query": SEARCH_QUERY, "display": DISPLAY_COUNT, "page": page, "type": "JSON"}
            try:
                response = await client.get(PRECEDENT_LIST_URL, params=params, timeout=30)
                response.raise_for_status()
                precedents, current_total = parse_precedent_list(response.text)
                
                if page == 1:
                    total_count = current_total
                    if total_count == 0:
                        print("검색 결과 없음. 검색어나 API 키를 확인하세요.")
                        return
                    print(f"총 {total_count}개의 판례 발견.")
                    progress_bar = tqdm_asyncio(total=total_count, desc="판례 목록 수집 중")

                if not precedents:
                    if progress_bar and progress_bar.n < progress_bar.total:
                        progress_bar.update(progress_bar.total - progress_bar.n)
                    break
                
                all_precedents.extend(precedents)
                if progress_bar:
                    progress_bar.update(len(precedents))
                
                if len(all_precedents) >= total_count:
                    break
                
                page += 1
                await asyncio.sleep(0.5) # API 서버 부하 감소

            except Exception as e:
                print(f"판례 목록 요청 실패 (페이지 {page}): {e}")
                break
    
    if progress_bar:
        progress_bar.close()

    if not all_precedents:
        print("수집된 판례가 없습니다.")
        return

    print(f"\n총 {len(all_precedents)}개의 판례 정보 수집 완료.")

    try:
        # 데이터프레임의 모든 값을 문자열로 변환
        df = pl.DataFrame(all_precedents).select([pl.all().cast(pl.Utf8)])
        df.write_parquet(OUTPUT_PARQUET_FILE)
        print(f"\n총 {len(df)}개 판례 목록을 '{OUTPUT_PARQUET_FILE}'에 저장했습니다.")
    except Exception as e:
        print(f"\nParquet 파일 저장 오류: {e}")

    print("\n" + "=" * 50)
    print("크롤링 완료!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

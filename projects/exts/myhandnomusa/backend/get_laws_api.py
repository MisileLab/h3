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
OUTPUT_PARQUET_FILE = "korean_labor_laws.parquet"
# 한 페이지에 요청할 법령 수 (최대 100)
DISPLAY_COUNT = 100
MAX_CONCURRENT_REQUESTS = 5  # 동시 요청 수 (API 서버 정책에 따라 조절)

# --- API 정보 ---
# 법령 목록 조회 API URL (제공된 문서 기준)
LAW_LIST_URL = "http://www.law.go.kr/DRF/lawSearch.do"
LAW_DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"

def parse_law_list(json_data):
    """법령 목록 JSON 응답에서 법령 리스트를 추출합니다."""
    try:
        data = json.loads(json_data)
        # API 응답은 'LawSearch' 키 아래에 중첩되어 있을 수 있습니다.
        if "LawSearch" in data:
            law_search_data = data["LawSearch"]
            items = law_search_data.get("law", [])
            total_count = int(law_search_data.get("totalCnt", "0"))
            return items, total_count
        # 일부 오류 응답은 최상위 레벨에 있습니다.
        elif "ERROR" in data:
            print(f"API 오류: {data['ERROR']['MESSAGE']}")
            return [], 0
        elif "faultInfo" in data:
            print(f"API 오류: {data['faultInfo']['message']}")
            return [], 0
        else:
            # 'law' 키가 최상위 레벨에 있는 원래의 경우도 처리합니다.
            if "law" in data:
                items = data["law"]
                total_count = int(data.get("totalCnt", "0"))
                return items, total_count
            else:
                print(f"알 수 없는 오류 또는 데이터 없음: {json_data}")
                return [], 0
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}\n데이터: {json_data}")
        return [], 0

def parse_law_detail(json_data):
    """법령 본문 JSON 응답에서 주요 텍스트 정보를 하나의 문자열로 결합합니다."""
    try:
        data = json.loads(json_data)
        texts = []

        reason = data.get("제개정이유내용")
        if reason: texts.append(f"[제개정 이유]\n{reason}\n")

        articles = data.get("조문", [])
        article_texts = [f"{a.get('조문제목', '')}\n{a.get('조문내용', '')}" for a in articles]
        if article_texts: texts.append("[조문]\n" + "\n\n".join(article_texts))

        addenda = data.get("부칙", [])
        addenda_texts = [item.get("부칙내용", "") for item in addenda]
        if addenda_texts: texts.append("[부칙]\n" + "\n\n".join(addenda_texts))
            
        return "\n".join(texts) if texts else "상세 내용 없음"

    except json.JSONDecodeError:
        return "상세 내용 파싱 오류"
    except Exception:
        try:
            error_data = json.loads(json_data)
            if "ERROR" in error_data: return f"API 오류: {error_data['ERROR']['MESSAGE']}"
            if "faultInfo" in error_data: return f"API 오류: {error_data['faultInfo']['message']}"
        except: pass
        return "상세 내용 파싱 중 알 수 없는 오류"

async def fetch_law_details(client, law_item, pbar):
    """법령 ID를 사용해 상세 정보를 가져와 원본 딕셔너리에 추가합니다."""
    law_id = law_item.get("법령ID")
    if not law_id:
        pbar.update(1)
        law_item['본문'] = '법령ID 없음'
        return law_item

    params = {"OC": API_KEY, "target": "law", "ID": law_id, "type": "JSON"}
    try:
        response = await client.get(LAW_DETAIL_URL, params=params, timeout=30)
        response.raise_for_status()
        law_item['본문'] = parse_law_detail(response.text)
    except httpx.RequestError as e:
        law_item['본문'] = f"상세 정보 요청 실패: {e}"
    finally:
        pbar.update(1)
        return law_item

async def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("법제처 Open API 기반 법령 및 본문 크롤링 스크립트")
    print("=" * 50)

    if API_KEY == "YOUR_API_KEY":
        print("오류: API 키 미설정. 환경 변수 'LAW_API_KEY'를 설정하세요.")
        return

    all_laws = []
    page = 1
    total_count = 0

    async with httpx.AsyncClient() as client:
        print(f"'{SEARCH_QUERY}' 키워드로 전체 법령 목록을 가져오는 중...")
        while True:
            params = {"OC": API_KEY, "target": "law", "query": SEARCH_QUERY, "display": DISPLAY_COUNT, "page": page, "type": "JSON"}
            try:
                response = await client.get(LAW_LIST_URL, params=params, timeout=30)
                response.raise_for_status()
                laws, current_total = parse_law_list(response.text)
                if page == 1:
                    total_count = current_total
                    if total_count == 0:
                        print("검색 결과 없음. 검색어나 API 키를 확인하세요.")
                        return
                    print(f"총 {total_count}개의 법령 발견.")
                if not laws: break
                all_laws.extend(laws)
                print(f"페이지 {page}: {len(laws)}개 수집 (총 {len(all_laws)}/{total_count})")
                if len(all_laws) >= total_count: break
                page += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"법령 목록 요청 실패 (페이지 {page}): {e}")
                break

        if not all_laws:
            print("수집된 법령이 없습니다.")
            return

        print(f"\n총 {len(all_laws)}개의 법령 상세 정보 수집 시작...")
        progress_bar = tqdm_asyncio(total=len(all_laws), desc="법령 상세 정보 수집 중")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def fetch_detail_for_item(item):
            async with semaphore:
                return await fetch_law_details(client, item, progress_bar)

        tasks = [fetch_detail_for_item(item) for item in all_laws]
        augmented_results = await asyncio.gather(*tasks)
        progress_bar.close()

        if augmented_results:
            try:
                df = pl.DataFrame(augmented_results)
                df.write_parquet(OUTPUT_PARQUET_FILE)
                print(f"\n총 {len(df)}개 법령과 본문을 '{OUTPUT_PARQUET_FILE}'에 저장했습니다.")
            except Exception as e:
                print(f"\nParquet 파일 저장 오류: {e}")
        else:
            print("\n수집된 상세 정보가 없어 파일을 저장하지 않았습니다.")

    print("\n" + "=" * 50)
    print("크롤링 완료!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

from requests import get

print(get("https://open.neis.go.kr/hub/mealServiceDietInfo?ATPT_OFCDC_SC_CODE=B10&SD_SCHUL_CODE=7010536&TYPE=json&MLSV_FROM_YMD=20230310&MLSV_TO_YMD=20240310").json())

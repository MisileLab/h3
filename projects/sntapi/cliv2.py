from pypdf import PdfReader
from pprint import PrettyPrinter
reader = PdfReader("a.pdf")
page = reader.pages[0]
pp = PrettyPrinter(indent=4)
sub = []
for i in page.extract_text().split("\n")[1:-1]:
    if i.startswith("※ 상기식단은 식자재 수급 사정에 의해 변동될 수 도 있습니다.☞식단관련 공지사항은 본교 홈페이지 및 학생식당에서 확인할 수 있습니다"):
        break
    sub.append(i)
sub = [i.split(" ") for i in sub]
pp.pprint(sub)
asd = {}
curnum = -1
for i in sub:
    lst = list(sorted(i, key=len))
    print(lst)


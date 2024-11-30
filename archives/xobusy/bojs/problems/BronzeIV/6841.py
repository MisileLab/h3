from sys import stdin

a = {
  "CU": "see you",
  ":-)": "I’m happy",
  ":-(": "I’m unhappy",
  ";-)": "wink",
  ":-P": "stick out my tongue",
  "(~.~)": "sleepy",
  "TA": "totally awesome",
  "CCC": "Canadian Computing Competition",
  "CUZ": "because",
  "TY": "thank-you",
  "YW": "you’re welcome",
  "TTYL": "talk to you later"
}

for i in stdin:
  i = i.strip('\n')
  if i == "":
    break
  try:
    print(a[i])
  except KeyError:
    print(i)
  if i == "TTYL":
    break

a = ""

try:
  while True:
    title = input("title: ")
    eurl = input("english url: ")
    kurl = input("korean url: ")
    tlang = input("if original is korean, please input anything: ")
    if tlang.strip() == "":
      tlang = "en"
    else:
      tlang = "kr"
    a += f"""- {title}
   - [{"Original" if tlang == "en" else "Translated"}(en)]({eurl})
   - [{"Original" if tlang == "kr" else "Translated"}(kr)]({kurl})
"""
except KeyboardInterrupt:
  print("---------")
  print(a)


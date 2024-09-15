a = ""
n = 0

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
    n += 1
except KeyboardInterrupt:
  if n == 1:
    a = '\n' + '\n'.join(i[5:] + '\\' for i in (a.splitlines()[1:]))
    a = a.strip('\\')
  else:
    a = '\n' + a
  print(a)

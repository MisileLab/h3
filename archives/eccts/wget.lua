local content = http.get(arg[1]).readAll()
fs.delete(arg[2])
local file = fs.open(arg[2], "w")
file.write(content)

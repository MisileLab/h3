
if fs.isDir("/.linoxos") == false then
    fs.makeDir("/.linoxos")
end
if fs.isDir("/.linoxs/packages") then
    fs.makeDir("/.linoxos/packages")
end

if arg[1] == "install" then
    table.remove(arg, 1)
    for _, v in ipairs(arg) do
        print("Installing " .. v)
    end
elseif arg[1] == "remove" then
    table.remove(arg, 1)
    for _, v in ipairs(arg) do
        print("Removing " .. v)
    end
end

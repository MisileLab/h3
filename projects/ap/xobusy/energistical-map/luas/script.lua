local _last = 0
local a = peripheral.wrap("back")

while true do
    if _last ~= a.getLastInput() then
        term.clear()
        term.setCursorPos(1,1)
        io.write(a.getLastInput())
        _last = a.getLastInput()
    end
    sleep(0.5)
end

local wsurl = "ws://localhost:1000"
local ws = http.websocket(wsurl)
local back = peripheral.wrap("back")

while true do
    ws.send(textutils.serialiseJSON({
        type = "induction",
        data = {
            current = back.getEnergy(),
            max = back.getMaxEnergy(),
            percent = back.getEnergyFilledPercentage(),
            input = back.getLastInput(),
            output = back.getLastOutput()
        }
    }))
    sleep(0.05)
end

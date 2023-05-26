local wsurl = "ws://localhost:1000"
local ws = http.websocket(wsurl)
local modem = peripheral.find("modem")

while true do
    ws.send(textutils.serializeJSON({
        type = "induction_redstone",
        data = {
            pneu = modem.callRemote("computer_0", "getRedstone")
        }
    }))
end

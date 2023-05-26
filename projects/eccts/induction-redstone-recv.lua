local wsurl = "ws://localhost:1000"
local ws = http.websocket(wsurl)
local modem = peripheral.find("modem")

while true do
    local msg = textutils.unserialiseJSON(ws.receive())
    if msg['type'] == 'induction' then
        modem.callRemote("computer_0", "setRedstone", msg['data']['pneu'])
    end
end

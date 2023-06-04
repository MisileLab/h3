local wsurl = "ws://localhost:1000"
local ws = http.websocket(wsurl)
local modem = peripheral.find("modem")
local map_recv = {
    pneu = "computer_0"
}

while true do
    local msg = textutils.unserialiseJSON(ws.receive())
    if msg['type'] == 'induction_redstone' then
        modem.callRemote(map_recv[msg['redstone_type']], "setRedstone", msg['redstone_value'])
    end
end

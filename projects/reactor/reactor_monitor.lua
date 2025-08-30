-- Reactor Monitoring Script for ComputerCraft
-- Communicates with FastAPI server via WebSocket

-- Check for required components
if not peripheral.wrap("fissionReactorLogicAdapter_0") then
  error("Fission reactor not found")
end

-- Configuration
local SERVER_URL = "ws://localhost:8765/ws/computercraft/1"
local UPDATE_INTERVAL = 1 -- seconds

-- Get reactor component
local reactor = peripheral.wrap("fissionReactorLogicAdapter_0")

-- Connect to WebSocket server
local ws, err = http.websocket(SERVER_URL)
if not ws then
  error("Failed to connect to server: " .. tostring(err))
end

print("Connected to reactor monitoring server")

-- Function to get reactor data
local function getReactorData()
  local data = {
    temperature = reactor.getTemperature() or 0,
    fuel_level = ((reactor.getFuel()["amount"] or 0) / (reactor.getFuelCapacity() or 1)) * 100,
    coolant_level = ((reactor.getCoolant()["amount"] or 0) / (reactor.getCoolantCapacity() or 1)) * 100,
    waste_level = ((reactor.getWaste()["amount"] or 0) / (reactor.getWasteCapacity() or 1)) * 100,
    status = reactor.getStatus() or false,
    burn_rate = reactor.getBurnRate() or 0,
    actual_burn_rate = reactor.getActualBurnRate() or 0,
    alert_status = 0 -- Will be calculated based on conditions
  }

  -- Calculate alert status
  -- 0 = normal, 1 = caution, 2 = danger
  if data.temperature > 1000 then
    data.alert_status = 2
  elseif data.temperature > 600 or data.coolant_level < 20 then
    data.alert_status = 1
  else
    data.alert_status = 0
  end

  return data
end

-- Function to send data to server
local function sendReactorData()
  local success, data = pcall(getReactorData)
  if success then
    local jsonData = textutils.serializeJSON(data)
    local ok, err = ws.send(jsonData)
    if not ok then
      print("Failed to send data: " .. tostring(err))
    else
      print("Data sent: " .. jsonData)
    end
  else
    print("Error getting reactor data: " .. tostring(data))
  end
end

-- Function to handle incoming commands
local function handleCommand(command)
  local success, data = pcall(textutils.unserialiseJSON, command)
  if success and type(data) == "table" then
    if data.cmd == "emergency_stop" then
      reactor.setEmergencyShutdown(true)
      print("Emergency stop activated")
    elseif data.cmd == "coolant_speed" and data.value then
      reactor.setCoolantSpeed(data.value)
      print("Coolant speed set to " .. data.value)
    end
  else
    print("Failed to parse command: " .. tostring(data))
  end
end

-- Main loop
local function main()
  print("Starting reactor monitoring...")

  while true do
    -- Send reactor data
    sendReactorData()

    -- Check for incoming commands
    local response = ws.receive(0.1) -- Non-blocking receive
    if response then
      handleCommand(response)
    end

    -- Wait for next update
    sleep(UPDATE_INTERVAL)
  end
end

-- Handle shutdown gracefully
local function shutdown()
  if ws then
    ws.close()
  end
  print("Reactor monitoring stopped")
end

-- Run main loop with error handling
local success, err = pcall(main)
if not success then
  print("Error: " .. tostring(err))
end

-- Cleanup
shutdown()


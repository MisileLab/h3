-- Reactor Monitoring Script for ComputerCraft
-- Communicates with FastAPI server via WebSocket

local component = require("component")
local computer = require("computer")
local event = require("event")
local websocket = require("websocket")
local serialization = require("serialization")

-- Check for required components
if not component.isAvailable("fusionReactor") then
  error("Fusion reactor not found!")
end

-- Configuration
local SERVER_URL = "ws://localhost:8765/ws/computercraft/1"
local UPDATE_INTERVAL = 1 -- seconds

-- Get reactor component
local reactor = component.fusionReactor

-- Connect to WebSocket server
local ws, err = websocket.connect(SERVER_URL)
if not ws then
  error("Failed to connect to server: " .. tostring(err))
end

print("Connected to reactor monitoring server")

-- Function to get reactor data
local function getReactorData()
  local data = {
    temperature = reactor.getTemperature() or 0,
    fuel_level = reactor.getFuelAmount() or 0,
    coolant_level = reactor.getCoolantAmount() or 0,
    waste_level = reactor.getWasteAmount() or 0,
    emergency_stop = reactor.isEmergencyShutdown() or false,
    coolant_speed = reactor.getCoolantSpeed() or 0,
    alert_status = 0 -- Will be calculated based on conditions
  }

  -- Calculate alert status
  -- 0 = normal, 1 = caution, 2 = danger
  if data.temperature > 85 then
    data.alert_status = 2
  elseif data.temperature > 60 or data.coolant_level < 20 then
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
    local jsonData = serialization.serialize(data)
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
  local success, data = pcall(serialization.unserialize, command)
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
    os.sleep(UPDATE_INTERVAL)
  end
end

-- Handle shutdown gracefully
local function shutdown()
  if ws then
    ws.close()
  end
  print("Reactor monitoring stopped")
end

-- Set up event handlers
event.listen("component_unavailable", function(_, componentType)
  if componentType == "fusionReactor" then
    error("Fusion reactor disconnected!")
  end
end)

-- Run main loop with error handling
local success, err = pcall(main)
if not success then
  print("Error: " .. tostring(err))
end

-- Cleanup
shutdown()


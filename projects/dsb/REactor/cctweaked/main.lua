-- Load REactor
print("Loading REactor...")

local ws = assert(http.websocket("http://localhost:8080"))

local function sender()
  while true do
    local ok, err = pcall(function()
      ws.send("Hello from REactor asynchronously!")
    end)
    if not ok then
      print("Error sending message: " .. err)
    end
    sleep(3) -- adjust interval as needed
  end
end

local function receiver()
  while true do
    local event, url, message, binary = os.pullEvent("websocket_message")
    if url == ws.url then
      print("Received asynchronously: " .. message)
    end
  end
end

parallel.waitForAll(sender, receiver)


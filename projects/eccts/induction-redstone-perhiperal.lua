local side = "left"

function setRedstone(value)
    return redstone.setAnalogOutput(side, value)
end

function getRedstone(value)
    return redstone.getAnalogOutput(side)
end

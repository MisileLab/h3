export interface Data {
  status: number
}

export const backendurl = import.meta.env.PROD ? (location.hostname.includes(".onion") ? "http://2vbh4msbvvbzo6wlzzsj3zeztjbh4fubh6h2lhqdh5xgpjs2hs7js2yd.onion/noa/api" : "https://xml.misile.xyz/noa/api") : "http://127.0.0.1:8000/noa/api"
export const host = location.protocol + "//" + location.host;
export const schale_url = location.hostname.includes(".onion") ? "http://2vbh4msbvvbzo6wlzzsj3zeztjbh4fubh6h2lhqdh5xgpjs2hs7js2yd.onion/schale/f" : "https://xml.misile.xyz/schale/f"
export function setCookie(name: string, value: string, days: number) {
  var expires = "";
  if (days) {
    var date = new Date();
    date.setTime(date.getTime() + (days*24*60*60*1000));
    expires = "; expires=" + date.toUTCString();
  }
  document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}
export function getCookie(name: string) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(';');
  for(var i=0;i < ca.length;i++) {
    var c = ca[i];
    while (c.charAt(0)==' ') c = c.substring(1,c.length);
    if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
  }
  return null;
}

export default function statusCheck(r: Data) {
  if (r.status >= 200 && r.status < 400) {return false;}
  if (r.status === 429) {
    alert('Please try again after some time');
  } else if (r.status == 403) {
    alert('Auth failed')
  } else {
    alert(`Error happend: status code -> ${r.status}`);
  }
  return true;
}


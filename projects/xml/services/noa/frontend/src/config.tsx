export interface Data {
  status: number
}

export const backendurl = import.meta.env.PROD ? (!location.hostname.includes(".onion") ? "https://noa.misile.xyz" : "onionurl") : "http://127.0.0.1:8000"
export const host = location.protocol + "//" + location.host;
export const schale_url = !location.hostname.includes(".onion") ? "https://schale.misile.xyz" : "onionurl"

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


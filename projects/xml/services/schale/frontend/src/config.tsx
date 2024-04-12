export interface Data {
  status: number
}

export const backendurl = import.meta.env.PROD ? (!location.hostname.includes(".onion") ? "https://schale.misile.xyz" : "onionurl") : "http://127.0.0.1:10001"
export const host = location.protocol + "//" + location.hostname;

export default function statusCheck(r: Data) {
  if (r.status === 429) {
    alert('Please try again after some time');
    return true;
  } else if (r.status < 100 || r.status > 300) {
    alert(`Error happend: status code -> ${r.status}`);
    return true;
  }
  return false;
}


export interface Data {
  status: number
}

export const backendurl = import.meta.env.PROD ? (location.hostname.includes(".onion") ? "http://2vbh4msbvvbzo6wlzzsj3zeztjbh4fubh6h2lhqdh5xgpjs2hs7js2yd.onion/schale/api" : "https://xml.misile.xyz/schale/api") : "http://127.0.0.1:10001/schale/api"
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

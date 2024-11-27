export interface Signer {
  name: string
  email: string
  message: string
  signature?: string
}

export interface Post {
  title: string
  tldr: string
  file: string
  signer: number
}

export const url = "https://misile.xyz/api"

async function fetchAPI<T>(
  path: string,
  headers: Record<string, string>,
  method: string = "GET",
  formdata: Record<string, string> | undefined = undefined
): Promise<T> {
  let fd = undefined;
  if (formdata !== undefined) {
    fd = new FormData();
    for (const i of Object.keys(formdata)) {
      fd.append(i, formdata[i])
    }
  }
  const header = new Headers()
  for (const i of Object.keys(headers)) {
    header.append(i, headers[i])
  }
  if (!path.startsWith("/")) {path = "/" + path;}
  return JSON.parse(await (await fetch(`${url}${path}`, {
    method: method,
    headers: headers,
    body: fd
  })).text()) as T
}

export async function getPost(name: string): Promise<Post> {
  return fetchAPI("info", {name: name})
}

export async function getSigner(
  name: string,
  name_signer: string
): Promise<Signer> {
  return fetchAPI("/info/signer", {name: name, "name-signer": name_signer})
}

export async function getSigners(name: string): Promise<Signer[]> {
  return fetchAPI("/info/signers", {name: name})
}

export async function confirm(
  name: string,
  name_signer: string,
  email: string,
  hash: string,
  message: string,
  signature: string
) {
  fetchAPI("/confirm", {
    name: name,
    "name-signer": name_signer,
    email: email,
    hash: hash,
    message: message,
    signature: signature
  })
}

import { fetchAPILow } from "../../../components/request"

export interface Signer {
  name: string
  email: string
  message: string
  signature?: string
}

export interface Post {
  name: string
  tldr: string
  file: string
  signer: number
}

async function fetchAPI<T>(
  path: string,
  headers: Record<string, string>,
  method: string = "GET",
  formdata: Record<string, string> | undefined = undefined
): Promise<T> {
  if (!path.startsWith("/")) {path = "/" + path;}
  return fetchAPILow(`/theresa/${path}`, headers, method, formdata)
}

export async function getPost(name: string): Promise<Post> {
  return fetchAPI("/info", {name: name})
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
  }, "POST")
}

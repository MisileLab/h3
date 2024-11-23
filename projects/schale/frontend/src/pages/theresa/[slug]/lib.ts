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

export const url = "http://127.0.0.1:8000/theresa"

async function fetchAPI<T>(path: string, headers: Record<string, unknown>): Promise<T> {
  // TODO add form data
  if (!path.startsWith("/")) {path = "/" + path;}
  return JSON.parse(await (await fetch(`${url}${path}`, headers=headers)).text()) as T
}

export async function getPost(name: string): Promise<Post> {
  return fetchAPI("info", {name: name})
}

export async function getSigner(name: string, signer_name: string): Promise<Signer> {
  return fetchAPI("/info/signer", {name: name, "signer-name": signer_name})
}

export async function getSigners(name: string): Promise<Signer[]> {
  return fetchAPI("/info/signers", {name: name})
}

// TODO remove after backend endpoint wrapped (only get wrapped for now)
export const f: Record<string, Post> = {
  pointer: {
    title: "Pointer",
    tldr: "Pointer that pointing to ideal",
    file: "https://github.com/MisileLab/h3/tree/main/projects/schale",
    signers: [{
      name: "Longnames",
      email: "longnames@duck.com",
      message: ""
    }, {
      name: "Misile",
      email: "misile@duck.com",
      message: "",
      signature: signature
    }]
  }
}

export class statusError extends Error {
  constructor(status: number, ...args: ErrorOptions[]) {
    const message = `status error: ${status}`;
    super(message, ...args);
    this.message = message
  }
}

export async function fetchAPILow<T>(
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
  const f = await fetch(path, {method: method, headers: headers, body: fd})
  const status = f.clone().status;
  const ok = f.clone().ok;
  if (!ok) {throw new statusError(status)}
  return JSON.parse(await f.text()) as T
}

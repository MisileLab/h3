export enum DataType {
  blog = "blog",
  news = "news"
}

export function getTextContent(h: HTMLElement) {
  return h.innerText == "" && h.textContent !== null ? h.textContent : h.innerText;
}

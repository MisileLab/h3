import { atom } from "nanostores"

export const path = atom("")
export const setPath = (newPath: string) => path.set(newPath)

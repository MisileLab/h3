import {z, defineCollection} from 'astro:content'

const urandom = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    createdDate: z.number(),
    tldr: z.string()
  })
})

export const collection = {
  urandom: urandom
}

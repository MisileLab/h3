import type { ColumnDef } from "@tanstack/react-table";

export type Comment = {
  author_name: string;
  content: string;
  comment_url: string;
  is_bot: boolean;
}

export const columns: ColumnDef<Comment>[] = [
  {
    header: "Author Name",
    accessorKey: "author_name"
  }, {
    header: "Content",
    accessorKey: "content"
  }, {
    header: "Comment URL",
    accessorKey: "comment_url"
  }, {
    header: "Is Bot",
    accessorKey: "is_bot"
  }
]

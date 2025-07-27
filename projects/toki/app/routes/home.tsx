import type { Route } from "./+types/home";
import { columns, type Comment } from "~/components/table/columns";
import { DataTable } from "~/components/table/data-table";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "noMoreSpam" },
    { name: "description", content: "remove spam from your youtube comments" },
  ];
}

export const comments: Comment[] = [
  {
    author_name: "John Doe",
    content: "This is a comment",
    comment_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    is_bot: false,
  },
  {
    author_name: "Jane Doe",
    content: "This is a comment",
    comment_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    is_bot: true,
  },
]

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <DataTable columns={columns} data={comments} />
    </div>
  );
}

import Link from "next/link"
import { CardTitle, CardHeader, CardContent, Card } from "@/components/ui/card"
import { existsSync, readFileSync, writeFileSync } from "fs"

export function Main() {
  if (!existsSync("data.json")) {
    writeFileSync("data.json", `
    {
      "resnum": {"recscan": 0, "vulfound": 0},
      "scans": []
    }
    `, {encoding: "utf-8"});
  }
  const data = JSON.parse(readFileSync("data.json", "utf-8"));
  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between h-16 px-6 shadow-sm bg-white dark:bg-gray-800">
        <div className="flex items-center space-x-4">
          <svg
            className=" h-6 w-6 text-blue-500 dark:text-blue-300"
            fill="none"
            height="24"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            viewBox="0 0 24 24"
            width="24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <span className="text-lg font-semibold">SecureScan</span>
          <nav className="hidden lg:flex space-x-2">
            <Link
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white"
              href="/"
            >
              Home
            </Link>
            <Link
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white"
              href="/newScan"
            >
              New Scan
            </Link>
            {/*<Link
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white"
              href="#"
            >
              Reports
            </Link>
            <Link
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white"
              href="#"
            >
              Settings
            </Link>*/}
          </nav>
        </div>
        {/*<div className="flex items-center space-x-2">
          <span className="text-gray-500 dark:text-gray-400">User123</span>
          <Link className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white" href="#">
            Logout
          </Link>
        </div>*/}
      </header>
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-64 bg-gray-50 dark:bg-gray-900 overflow-y-auto">
          <nav className="mt-5 px-2 space-y-1">
            <Link
              className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-200 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white"
              href="/newScan"
            >
              New Scan
            </Link>
            <Link
              className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-200 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white"
              href="/savedScans"
            >
              Saved Scans
            </Link>
            {/*<Link
              className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-200 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white"
              href="#"
            >
              Scheduled Scans
            </Link>
            <Link
              className="flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-200 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white"
              href="#"
            >
              Scan History
            </Link>*/}
          </nav>
        </aside>
        <main className="flex-1 p-4 overflow-y-auto">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Recent Scans</CardTitle>
                <svg
                  className=" w-4 h-4 text-zinc-500 dark:text-zinc-400"
                  fill="none"
                  height="24"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  width="24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                  <path d="M21 3v5h-5" />
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                  <path d="M8 16H3v5" />
                </svg>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{data.resnum.recscan}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Vulnerabilities Found</CardTitle>
                <svg
                  className=" w-4 h-4 text-zinc-500 dark:text-zinc-400"
                  fill="none"
                  height="24"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  width="24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                  <path d="M12 9v4" />
                  <path d="M12 17h.01" />
                </svg>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{data.resnum.vulfound}</div>
              </CardContent>
            </Card>
            {/*<Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Security Recommendations</CardTitle>
                <svg
                  className=" w-4 h-4 text-zinc-500 dark:text-zinc-400"
                  fill="none"
                  height="24"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  width="24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4" />
                  <path d="M12 8h.01" />
                </svg>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">5</div>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">new recommendations</p>
              </CardContent>
            </Card>*/}
          </div>
        </main>
      </div>
    </div>
  )
}

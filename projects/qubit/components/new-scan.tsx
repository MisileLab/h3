'use client'

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { exec } from "child_process";
import { useState } from "react";
import { existsSync, readFileSync, rmSync } from "fs";

export function NewScan() {
  const [path, setPath] = useState("");
  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between h-16 px-6 shadow-sm bg-white dark:bg-gray-800">
        <div className="flex items-center space-x-4">
          <IconShieldCheck className="h-6 w-6 text-blue-500 dark:text-blue-300" />
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
            </Link>*/}
          </nav>
        </aside>
        <main className="flex-1 p-4 overflow-y-auto">
          <form>
            <div className="mb-4">
              <label className="text-sm text-gray-600 dark:text-gray-400" htmlFor="scanPath">
                Scan Path
              </label>
              <input
                className="mt-1 block w-full px-4 py-2 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 dark:bg-gray-700 dark:text-white"
                id="scanPath"
                type="text"
                onChange={(self)=>{setPath(self.target.value)}}
              />
            </div>
            <Button
              className="w-full py-2 px-4 rounded-md bg-blue-500 text-white hover:bg-blue-600 dark:bg-blue-400 dark:hover:bg-blue-500"
              type="submit"
              onClick={()=>{
                if (path == "") {
                  return;
                }
                const data = JSON.parse(readFileSync("data.json", "utf-8").toString());
                data.resnum.recscan++;
                if (existsSync("report.json") {rmSync("report.json");}
                exec(`gitleaks detect ${path} --no-git --report-format json --report-path report.json`);
                const vuls = [];
                const _data = JSON.parse(readFileSync("report.json", "utf-8").toString());
                for (const i of _data) {
                  data.resnum.vulfound++;
                  vuls.push({
                    name: "Secret found",
                    Line: i["StartLine"],
                    Column: i["StartColumn"],
                    description: i["Description"]
                  })
                }
                rmSync("report.json");
                exec(`snyk code test --json-file-output=report.json`);
                const _data2 = JSON.parse(readFileSync("report.json", "utf-8").toString());
                // only uses location in runs if location exist, if location not exist, dont use it
              }}>
              Start Scan
            </Button>
          </form>
        </main>
      </div>
    </div>
  )
}


function IconShieldCheck(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  )
}

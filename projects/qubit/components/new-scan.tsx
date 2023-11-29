'use client'

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Command } from '@tauri-apps/api/shell'
import {exists, BaseDirectory, writeTextFile, readTextFile, removeFile} from '@tauri-apps/api/fs';
import {appCacheDir, join} from "@tauri-apps/api/path";
import {useState} from "react";

export function NewScan() {
    const [path, setPath] = useState("");
  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between h-16 px-6 shadow-sm bg-white dark:bg-gray-800">
        <div className="flex items-center space-x-4">
          <IconShieldCheck className="h-6 w-6 text-blue-500 dark:text-blue-300" />
            <Link href="/"><span className="text-lg font-semibold">SecureScan</span></Link>
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
                onChange={(self) => {setPath(self.target.value);}}
              />
            </div>
            <Button
              className="w-full py-2 px-4 rounded-md bg-blue-500 text-white hover:bg-blue-600 dark:bg-blue-400 dark:hover:bg-blue-500"
              type="submit"
              onClick={()=>{
                  async function b() {
                      if (path == "") {
                          return;
                      }
                      console.log(path);
                      const data = JSON.parse(readTextFile("data.json", { dir: BaseDirectory.Data }).toString());
                      data.resnum.recscan++;
                      const reportFile = await join(await appCacheDir(), "report.json")
                      if (!await exists("report.json", {dir: BaseDirectory.AppCache})) {await removeFile("report.json", { dir: BaseDirectory.AppCache });}
                      await new Command('run-gitleaks', ['gitleaks', 'detect', reportFile, '--no-git', '--report-format', 'json', '--report-path', 'report.json']).spawn();
                      const vuls = [];
                      const _data = JSON.parse(readTextFile("report.json", { dir: BaseDirectory.AppCache }).toString());
                      for (const i of _data) {
                          data.resnum.vulfound++;
                          vuls.push({
                              name: `secret found on ${i["File"]}`,
                              Line: i["StartLine"],
                              Column: i["StartColumn"],
                              description: i["Description"]
                          })
                      }
                      await removeFile("report.json", { dir: BaseDirectory.AppCache });
                      await new Command('run-snyk', ['snyk', 'code', 'test', reportFile, '--json-file-output=report.json']).spawn();
                      const _data2 = JSON.parse(readTextFile("report.json", { dir: BaseDirectory.AppCache }).toString());
                      for (const i of _data2.runs.tool.driver.results) {
                          data.resnum.vulfound++;
                          vuls.push({
                              name: `found ${i.ruleId} in ${i.locations[0].physicalLocation.artifactLocation.uri}`,
                              description: i.message.text,
                              Line: i.locations[0].physicalLocation.region.startLine,
                              Column: i.locations[0].physicalLocation.artifactLocation.endLine
                          })
                      }
                      data.scans.push({
                          "path": path,
                          "leaks": vuls
                      });
                      writeTextFile("data.json", JSON.stringify(data), { dir: BaseDirectory.AppCache })
                  }
                  b();
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

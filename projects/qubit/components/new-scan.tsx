'use client'

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Command } from '@tauri-apps/api/shell'
import {exists, BaseDirectory, writeTextFile, readTextFile, removeFile} from '@tauri-apps/api/fs';
import {appCacheDir, join} from "@tauri-apps/api/path";
import {useState} from "react";
import {type} from "@tauri-apps/api/os";

export function NewScan() {
    const [path, setPath] = useState("");
  return (
    <div className="flex flex-col h-screen">
      <header className="flex items-center justify-between h-16 px-6 shadow-sm bg-white dark:bg-gray-800">
        <div className="flex items-center space-x-4">
          <IconShieldCheck className="h-6 w-6 text-blue-500 dark:text-blue-300" />
            <Link href="/"><span className="text-lg font-semibold">Qubit</span></Link>
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
              type="button"
              onClick={()=>{
                  async function b() {
                      if (path == "") {
                          return;
                      }
                      console.log(path);
                      const data = JSON.parse(await readTextFile("data.json", { dir: BaseDirectory.AppData }));
                      data.resnum.recscan++;
                      data.scans.push({
                        "path": path,
                        "leaks": []
                      });
                      const length = data.scans.length-1;
                      const snykReportFile = await join(await appCacheDir(), "report_snyk.json")
                      const gitleaksReportFile = await join(await appCacheDir(), "report_gitleaks.json")
                      const cmd = new Command('run-gitleaks', ['detect', '--no-git', '--report-format', 'json', '--report-path', gitleaksReportFile, path, '-v'])
                      cmd.on("close", async (_)=>{
                        console.log(await exists("report_gitleaks.json", {dir: BaseDirectory.AppCache}))
                        if (!await exists("report_gitleaks.json", {dir: BaseDirectory.AppCache})) {return;}
                        const _data = JSON.parse(await readTextFile("report_gitleaks.json", { dir: BaseDirectory.AppCache }));
                        for (const i of _data) {
                            data.resnum.vulfound++;
                            data.scans[length].leaks.push({
                                name: `secret found on ${i["File"]}`,
                                Line: i["StartLine"],
                                Column: i["StartColumn"],
                                description: i["Description"]
                            })
                        }
                        console.log(data, _data);
                        await writeTextFile("data.json", JSON.stringify(data), { dir: BaseDirectory.AppData })
                        await removeFile("report_gitleaks.json", { dir: BaseDirectory.AppCache });
                      })
                      cmd.stdout.on("data", a=>console.log(a))
                      cmd.stderr.on("data",a=>console.warn(a))
                      await cmd.spawn();
                      let cmd2 = new Command('run-snyk', ['code', 'test', '-d', path, `--json-file-output=${snykReportFile}`]);
                      if (await type() == "Windows_NT") {
                          cmd2 = new Command("run-shell", ['/C', 'snyk', 'code', 'test', '-d', path, `--json-file-output=${snykReportFile}`])
                      }
                      cmd2.on("close", async (_) => {
                        console.log(await exists("report_snyk.json", {dir: BaseDirectory.AppCache}))
                        if (!await exists("report_snyk.json", {dir: BaseDirectory.AppCache})) {return;}
                        const _data2 = JSON.parse(await readTextFile("report_snyk.json", { dir: BaseDirectory.AppCache }))
                        for (const i of _data2.runs[0].results) {
                          data.resnum.vulfound++;
                          data.scans[length].leaks.push({
                              name: `found ${i.ruleId} in ${i.locations[0].physicalLocation.artifactLocation.uri}`,
                              description: i.message.text,
                              Line: i.locations[0].physicalLocation.region.startLine,
                              Column: i.locations[0].physicalLocation.region.startColumn
                          })
                      }
                      console.log(data, _data2);
                      await writeTextFile("data.json", JSON.stringify(data), { dir: BaseDirectory.AppData })
                      })
                      cmd2.stdout.on("data", a=>console.log(a))
                      cmd2.stderr.on("data",e=>console.warn(e))
                      await cmd2.spawn()
                      console.log(data);
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


function IconShieldCheck(props: any) {
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

'use client'

import React from 'react';
import Link from "next/link"
import { SelectValue, SelectTrigger, SelectLabel, SelectItem, SelectGroup, SelectContent, Select } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { ScanRes } from './scanres';

export function Scan() {
  const [openState, setOpenState] = React.useState(false);
  
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
          <h1 className="text-2xl font-semibold mb-4">Saved Scans</h1>
          <div className="bg-white dark:bg-gray-800 rounded-md shadow-sm p-6 mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-lg font-medium">Scan #1</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">10/11/2023</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <p>Path: /user/documents</p>
              <p>Status: Completed</p>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select a leak" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Leaks</SelectLabel>
                    <SelectItem value="leak1">Leak 1</SelectItem>
                    <SelectItem value="leak2">Leak 2</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              <Button className="mt-2" variant="outline" onClick={()=>{setOpenState(true);}}>
                Show Result
              </Button>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-md shadow-sm p-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-lg font-medium">Scan #2</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">09/11/2023</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <p>Path: /user/downloads</p>
              <p>Status: Completed</p>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select a leak" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Leaks</SelectLabel>
                    <SelectItem value="leak1">Leak 1</SelectItem>
                    <SelectItem value="leak2">Leak 2</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              <Button className="mt-2" variant="outline" onClick={()=>setOpenState(false)}>
                Show Result
              </Button>
            </div>
          </div>
        </main>
      </div>
      {/*<div className="fixed z-10 inset-0 overflow-y-auto">
        <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
          <div aria-hidden="true" className="fixed inset-0 transition-opacity">
            <div className="absolute inset-0 bg-gray-500 opacity-75" />
          </div>
          <span aria-hidden="true" className="hidden sm:inline-block sm:align-middle sm:h-screen">
            ​
          </span>
          <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="sm:flex sm:items-start">
                <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                  <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                    Scan Result
                  </h3>
                  <div className="mt-2">
                    <p className="text-sm text-gray-500">Vulnerability Name: XYZ</p>
                    <p className="text-sm text-gray-500">
                      Description: This vulnerability allows an attacker to execute arbitrary code.
                    </p>
                    <p className="text-sm text-gray-500">Line:Column - 10:35</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
              <Button
                className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm"
                variant="default"
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      </div>*/}
      {openState ? <ScanRes callback={()=>{setOpenState(false)}}/> : null}
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

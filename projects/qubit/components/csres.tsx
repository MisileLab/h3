'use client'

import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue
} from "@/components/ui/select";
import {Button} from "@/components/ui/button";
import React from "react";
import {ScanRes} from "@/components/scanres";

// @ts-ignore
export function ResScan({path, leaks, num}) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const [openState, setOpenState] = React.useState({
        "index": 0,
        "opened": false
    });
    const items = () => {
        const elists: any[] = [];
        leaks.forEach((e: any, i: number)=>{
            elists.push(<SelectItem value={i.toString()}>{e["name"]}</SelectItem>)
        })
        return elists;
    }
    return (<div className="bg-white dark:bg-gray-800 rounded-md shadow-sm p-6 mb-4">
        <div className="flex justify-between items-center mb-2">
            <span className="text-lg font-medium">Scan #{num}</span>
            <span className="text-sm text-gray-500 dark:text-gray-400">10/11/2023</span>
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400">
            <p>Path: {path}</p>
            <Select onValueChange={(s)=>{console.log(s);setOpenState({"index": Number(s), "opened": false})}}>
                <SelectTrigger>
                    <SelectValue placeholder="Select a leak" />
                </SelectTrigger>
                <SelectContent>
                    <SelectGroup className="bg-white">
                        <SelectLabel>Leaks</SelectLabel>
                        {items()}
                    </SelectGroup>
                </SelectContent>
            </Select>
            <Button className="mt-2" variant="outline" onClick={()=>{setOpenState({
                "index": openState["index"],
                "opened": leaks.length != 0
            });}}>
                Show Result
            </Button>
        </div>
        {openState["opened"] ? <ScanRes leak={leaks[openState["index"]]} callback={()=>{setOpenState({
            "index": openState["index"],
            opened: false
        })}}/> : null}
    </div>);
}

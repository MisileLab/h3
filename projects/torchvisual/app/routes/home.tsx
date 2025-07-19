import { ReactFlow, applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";
import { useState, useCallback } from "react";
import type { Route } from "./+types/home";
import "@xyflow/react/dist/style.css";
import { Select, SelectContent, SelectTrigger, SelectValue, SelectItem } from "~/components/ui/select";

interface Argument {
  name: string;
  type: string;
  description?: string;
}

interface Layer {
  name: string;
  arguments: Record<string, Argument>;
}

const Layers: Record<string, Layer> = {
  linear: {
    name: "Linear",
    arguments: {
      in_features: {
        name: "in_features",
        type: "number",
        description: "Size of each input sample.",
      },
      out_features: {
        name: "out_features",
        type: "number",
        description: "Size of each output sample.",
      },
      bias: {
        name: "bias",
        type: "boolean",
        description: "If set to false, the layer will not learn an additive bias.",
      },
    },
  }
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: "torchvisual" },
    { name: "description", content: "Preprocess, Train, Test. Replace these repeating steps." },
  ];
}

export default function Home() {
  const initialNodes = [
    { id: 'n1', position: { x: 0, y: 0 }, data: { label: 'Node 1' } },
    { id: 'n2', position: { x: 0, y: 100 }, data: { label: 'Node 2' } },
  ];
  const initialEdges = [{ id: 'n1-n2', source: 'n1', target: 'n2' }];

  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [selectedLayerType, setSelectedLayerType] = useState("linear");
 
  const onNodesChange = useCallback(
    // @ts-ignore - I gave up type annotation
    (changes) => setNodes((nodesSnapshot) => applyNodeChanges(changes, nodesSnapshot)),
    [],
  );
  const onEdgesChange = useCallback(
    // @ts-ignore - I gave up type annotation
    (changes) => setEdges((edgesSnapshot) => applyEdgeChanges(changes, edgesSnapshot)),
    [],
  );
  const onConnect = useCallback(
    // @ts-ignore - I gave up type annotation
    (params) => setEdges((edgesSnapshot) => addEdge(params, edgesSnapshot)),
    [],
  );
 
  return (
    <div className="h-screen w-screen flex flex-row bg-background">
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
        />
      </div>
      <div className="bg-overlay1 w-3/12 h-full flex flex-col gap-4 px-3 py-3">
        <h1 className="font-bold text-xl">Layer Type</h1>
        <Select onValueChange={(value) => setSelectedLayerType(value)} value={selectedLayerType}>
          <SelectTrigger className="w-full bg-white">
            <SelectValue placeholder="Linear" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="linear">Linear</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

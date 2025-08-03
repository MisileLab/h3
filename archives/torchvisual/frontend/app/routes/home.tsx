import { ReactFlow, applyNodeChanges, applyEdgeChanges, addEdge, type Node, type Edge } from "@xyflow/react";
import { useState, useCallback, useEffect } from "react";
import type { Route } from "./+types/home";
import "@xyflow/react/dist/style.css";
import { Select, SelectContent, SelectTrigger, SelectValue, SelectItem } from "~/components/ui/select";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "~/components/ui/context-menu";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";

interface Argument {
  name: string;
  type: string;
  description?: string;
}

interface Layer {
  name: string;
  arguments: Record<string, Argument>;
}

function useIsClient() {
  const [isClient, setIsClient] = useState(false);
  
  useEffect(() => {
    setIsClient(true);
  }, []);
  
  return isClient;
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: "torchvisual" },
    { name: "description", content: "Preprocess, Train, Test. Replace these repeating steps." },
  ];
}

export default function Home() {
  const layers: Record<string, Layer> = {
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

  const initialNodes: Node[] = [];
  const initialEdges: Edge[] = [];

  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [generating, isGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState("");
  const [selectedLayerType, setSelectedLayerType] = useState("linear");

  useEffect(() => {
    if (generating) {
      fetch('http://localhost:8000/generate', {
        method: 'POST'
      }).then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        response.text().then((data) => {
          console.log(data);
          setGeneratedText(data);
        })
      })
    }
  }, [generating]);
 
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

  const isClient = useIsClient();

  if (!isClient) {
    return <div></div>; // or a loading spinner
  }

  return (
    <div className="h-screen w-screen flex flex-row bg-background">
      <div className="flex-1">
        <ContextMenu>
          <ContextMenuTrigger>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              fitView
            />
          </ContextMenuTrigger>
          <ContextMenuContent>
            <ContextMenuItem onClick={() => {
              const newNode: Node = {
                id: `${nodes.length + 1}`,
                type: "default",
                position: { x: Math.random() * 300, y: Math.random() * 300 },
                data: {
                  label: layers[selectedLayerType].name,
                  arguments: layers[selectedLayerType].arguments,
                },
              };
              setNodes((nds) => nds.concat(newNode));
            }}>Add new nodes</ContextMenuItem>
            <ContextMenuItem onClick={() => {

            }}>Generate code</ContextMenuItem>
          </ContextMenuContent>
        </ContextMenu>
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
        <Label htmlFor="linearName">Layer Name</Label>
        <Input type="text" id="layerName" placeholder="Enter layer name" className="bg-white" />
        <Label htmlFor="linearInFeatures">In Features</Label>
        <Input type="number" id="linearInFeatures" placeholder="Enter in features" className="bg-white" />
        <Label htmlFor="linearOutFeatures">Out Features</Label>
        <Input type="number" id="linearOutFeatures" placeholder="Enter out features" className="bg-white" />
        <Label htmlFor="linearBias">Bias</Label>
        <Select onValueChange={(value) => console.log(value)} value="true">
          <SelectTrigger className="w-full bg-white">
            <SelectValue placeholder="true" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="true">True</SelectItem>
            <SelectItem value="false">False</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

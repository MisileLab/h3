import { ReactFlow, applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";
import { useState, useCallback } from "react";
import type { Route } from "./+types/home";
import "@xyflow/react/dist/style.css";

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
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      />
    </div>
  );
}

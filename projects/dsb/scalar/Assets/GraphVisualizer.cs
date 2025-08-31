using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using System.Linq;

namespace Scalar
{
    /// <summary>
    /// Visual representation of the exploration graph
    /// </summary>
    public class GraphVisualizer : MonoBehaviour
    {
        [Header("Visual Settings")]
        [SerializeField] private GameObject nodePrefab;
        [SerializeField] private GameObject connectionPrefab;
        [SerializeField] private Material hiddenNodeMaterial;
        [SerializeField] private Material visibleNodeMaterial;
        [SerializeField] private Material accessibleNodeMaterial;
        [SerializeField] private Material completedNodeMaterial;
        
        [Header("Node Colors by Type")]
        [SerializeField] private Color startNodeColor = Color.green;
        [SerializeField] private Color combatNodeColor = Color.red;
        [SerializeField] private Color resourceNodeColor = Color.yellow;
        [SerializeField] private Color eventNodeColor = Color.blue;
        [SerializeField] private Color dangerNodeColor = Color.magenta;
        [SerializeField] private Color repairNodeColor = Color.cyan;
        [SerializeField] private Color extractionNodeColor = Color.white;
        [SerializeField] private Color bossNodeColor = Color.black;
        [SerializeField] private Color shopNodeColor = Color.gray;
        [SerializeField] private Color restNodeColor = Color.green;
        
        [Header("Layout Settings")]
        [SerializeField] private float nodeScale = 1.0f;
        [SerializeField] private float connectionWidth = 0.1f;
        [SerializeField] private float heightOffset = 0.0f;
        [SerializeField] private float gridSpacing = 3.0f; // Increased from 2.0f for better visibility
        
        // Public property for other components to access grid spacing
        public float GridSpacing => gridSpacing;
        
        private GraphSystem graphSystem;
        private Dictionary<Vector2Int, GameObject> nodeObjects;
        private Dictionary<string, GameObject> connectionObjects;
        private Transform graphContainer;
        
        void Start()
        {
            graphSystem = FindFirstObjectByType<GraphSystem>();
            if (graphSystem == null)
            {
                Debug.LogError("GraphVisualizer: No GraphSystem found in scene!");
                return;
            }
            
            InitializeVisualizer();
            
            // Subscribe to graph events
            graphSystem.OnNodeDiscovered.AddListener(OnNodeDiscovered);
            graphSystem.OnNodeCompleted.AddListener(OnNodeCompleted);
            
            // Subscribe to incremental generation events
            if (graphSystem.GetType().GetField("OnNewChunkGenerated", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance) != null)
            {
                var onNewChunkGenerated = graphSystem.GetType().GetField("OnNewChunkGenerated", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance).GetValue(graphSystem) as UnityEvent<int>;
                if (onNewChunkGenerated != null)
                {
                    onNewChunkGenerated.AddListener(OnNewChunkGenerated);
                }
            }
        }
        
        void Update()
        {
            // Update graph visualization when it's ready
            if (graphSystem.IsGraphGenerated())
            {
                if (nodeObjects.Count == 0)
                {
                    CreateGraphVisualization();
                }
                else
                {
                    // Continuously check for newly visible nodes
                    UpdateVisibleNodes();
                }
            }
        }
        
        /// <summary>
        /// Initialize the visualizer
        /// </summary>
        private void InitializeVisualizer()
        {
            nodeObjects = new Dictionary<Vector2Int, GameObject>();
            connectionObjects = new Dictionary<string, GameObject>();
            
            // Create container for graph objects
            graphContainer = new GameObject("GraphContainer").transform;
            graphContainer.SetParent(transform);
        }
        
        /// <summary>
        /// Create the complete graph visualization
        /// </summary>
        private void CreateGraphVisualization()
        {
            if (graphSystem == null) return;
            
            // Create all nodes
            var allNodes = graphSystem.GetVisibleNodes();
            foreach (var node in allNodes)
            {
                CreateNodeVisual(node);
            }
            
            // Validate connections before creating visuals
            ValidateConnections();
            
            // Create connections
            CreateConnections();
            
            Debug.Log($"GraphVisualizer: Created visualization with {nodeObjects.Count} nodes");
        }
        
        /// <summary>
        /// Create visual representation of a node
        /// </summary>
        private void CreateNodeVisual(ExplorationNode node)
        {
            if (nodeObjects.ContainsKey(node.position)) return;
            
            Vector3 worldPos = GetNodeWorldPosition(node.position);
            Debug.Log($"GraphVisualizer: Creating node {node.nodeId} at grid position {node.position} -> world position {worldPos}");
            
            GameObject nodeObj;
            
            if (nodePrefab != null)
            {
                nodeObj = Instantiate(nodePrefab, worldPos, Quaternion.identity);
            }
            else
            {
                // Create default node if no prefab
                nodeObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                nodeObj.transform.position = worldPos;
            }
            
            nodeObj.transform.SetParent(graphContainer);
            nodeObj.transform.localScale = Vector3.one * nodeScale;
            nodeObj.name = $"Node_{node.type}_{node.position.x}_{node.position.y}";
            
            // Set node color based on type
            SetNodeColor(nodeObj, node);
            
            // Set node material based on state
            SetNodeMaterial(nodeObj, node);
            
            // Add node info component
            var nodeInfo = nodeObj.AddComponent<NodeInfo>();
            nodeInfo.Initialize(node);
            
            nodeObjects[node.position] = nodeObj;
            Debug.Log($"GraphVisualizer: Created node object {nodeObj.name} at {worldPos}");
        }
        
        /// <summary>
        /// Create visual connections between nodes
        /// </summary>
        private void CreateConnections()
        {
            var allNodes = graphSystem.GetVisibleNodes();
            Debug.Log($"GraphVisualizer: Creating connections for {allNodes.Count} nodes");
            
            foreach (var node in allNodes)
            {
                // Only create connections to adjacent nodes in the grid
                var adjacentPositions = GetAdjacentPositions(node.position);
                Debug.Log($"GraphVisualizer: Node {node.nodeId} at {node.position} has {adjacentPositions.Count} adjacent positions");
                
                foreach (var adjacentPos in adjacentPositions)
                {
                    // Check if there's actually a node at the adjacent position
                    var adjacentNode = graphSystem.GetNodeAt(adjacentPos);
                    if (adjacentNode != null)
                    {
                        Debug.Log($"GraphVisualizer: Found adjacent node {adjacentNode.nodeId} at {adjacentPos}");
                        
                        // Only create connection if both nodes are connected in the graph system
                        if (node.connectedNodeIds.Contains(adjacentNode.nodeId) && 
                            adjacentNode.connectedNodeIds.Contains(node.nodeId))
                        {
                            Debug.Log($"GraphVisualizer: Creating connection between {node.nodeId} and {adjacentNode.nodeId}");
                            CreateConnection(node, adjacentNode);
                        }
                        else
                        {
                            Debug.Log($"GraphVisualizer: Nodes {node.nodeId} and {adjacentNode.nodeId} are not connected in graph system");
                        }
                    }
                    else
                    {
                        Debug.Log($"GraphVisualizer: No node found at adjacent position {adjacentPos}");
                    }
                }
            }
            
            Debug.Log($"GraphVisualizer: Created {connectionObjects.Count} visual connections");
        }
        
        /// <summary>
        /// Validate that connections only exist between adjacent nodes
        /// </summary>
        private void ValidateConnections()
        {
            var allNodes = graphSystem.GetVisibleNodes();
            Debug.Log("GraphVisualizer: Validating connections...");
            
            foreach (var node in allNodes)
            {
                foreach (var connectedId in node.connectedNodeIds)
                {
                    var connectedNode = allNodes.FirstOrDefault(n => n.nodeId == connectedId);
                    if (connectedNode != null)
                    {
                        // Check if nodes are actually adjacent
                        var distance = Vector2Int.Distance(node.position, connectedNode.position);
                        if (distance > 1.1f) // Allow small floating point error
                        {
                            Debug.LogWarning($"GraphVisualizer: Invalid connection detected! Node {node.nodeId} at {node.position} is connected to {connectedNode.nodeId} at {connectedNode.position} (distance: {distance})");
                        }
                        else
                        {
                            Debug.Log($"GraphVisualizer: Valid connection: {node.nodeId} at {node.position} -> {connectedNode.nodeId} at {connectedNode.position} (distance: {distance})");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"GraphVisualizer: Node {node.nodeId} references non-existent connected node {connectedId}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Get adjacent grid positions (up, down, left, right)
        /// </summary>
        private List<Vector2Int> GetAdjacentPositions(Vector2Int pos)
        {
            var adjacent = new List<Vector2Int>();
            
            // Check adjacent positions (same logic as GraphSystem)
            Vector2Int[] directions = {
                new Vector2Int(1, 0),   // Right
                new Vector2Int(0, 1),   // Up
                new Vector2Int(0, -1),  // Down
                new Vector2Int(-1, 0)   // Left
            };
            
            foreach (var dir in directions)
            {
                Vector2Int checkPos = pos + dir;
                adjacent.Add(checkPos);
            }
            
            return adjacent;
        }
        
        /// <summary>
        /// Create a visual connection between two nodes
        /// </summary>
        private void CreateConnection(ExplorationNode nodeA, ExplorationNode nodeB)
        {
            string connectionId = GetConnectionId(nodeA, nodeB);
            if (connectionObjects.ContainsKey(connectionId)) return;
            
            Vector3 startPos = GetNodeWorldPosition(nodeA.position);
            Vector3 endPos = GetNodeWorldPosition(nodeB.position);
            
            Debug.Log($"GraphVisualizer: Creating connection from {nodeA.position} ({startPos}) to {nodeB.position} ({endPos})");
            
            GameObject connectionObj;
            
            if (connectionPrefab != null)
            {
                connectionObj = Instantiate(connectionPrefab);
            }
            else
            {
                // Create default connection using LineRenderer for better line visualization
                connectionObj = new GameObject($"Connection_{connectionId}");
                var lineRenderer = connectionObj.AddComponent<LineRenderer>();
                
                // Configure LineRenderer
                lineRenderer.material = GetConnectionMaterial(nodeA, nodeB);
                lineRenderer.startWidth = connectionWidth;
                lineRenderer.endWidth = connectionWidth;
                lineRenderer.positionCount = 2;
                lineRenderer.useWorldSpace = true;
                lineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lineRenderer.receiveShadows = false;
                
                // Set line positions
                lineRenderer.SetPosition(0, startPos);
                lineRenderer.SetPosition(1, endPos);
                
                // Ensure we have a material
                if (lineRenderer.material == null)
                {
                    // Create a default material if none is available
                    var defaultMaterial = new Material(Shader.Find("Sprites/Default"));
                    defaultMaterial.color = Color.white;
                    lineRenderer.material = defaultMaterial;
                }
            }
            
            connectionObj.transform.SetParent(graphContainer);
            
            // For LineRenderer, we don't need to handle positioning, scaling, or rotation
            // The LineRenderer handles everything automatically
            
            // Set connection material (only needed for prefab connections)
            if (connectionPrefab != null)
            {
                var renderer = connectionObj.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = GetConnectionMaterial(nodeA, nodeB);
                }
            }
            
            connectionObjects[connectionId] = connectionObj;
            Debug.Log($"GraphVisualizer: Created connection object {connectionId} from {startPos} to {endPos}");
        }
        
        /// <summary>
        /// Get world position for a grid position
        /// </summary>
        private Vector3 GetNodeWorldPosition(Vector2Int gridPos)
        {
            return new Vector3(gridPos.x * gridSpacing, heightOffset, gridPos.y * gridSpacing);
        }
        

        
        /// <summary>
        /// Get connection ID for two nodes
        /// </summary>
        private string GetConnectionId(ExplorationNode nodeA, ExplorationNode nodeB)
        {
            // Ensure consistent ordering
            if (string.Compare(nodeA.nodeId, nodeB.nodeId) < 0)
            {
                return $"{nodeA.nodeId}_{nodeB.nodeId}";
            }
            return $"{nodeB.nodeId}_{nodeA.nodeId}";
        }
        
        /// <summary>
        /// Set node color based on type
        /// </summary>
        private void SetNodeColor(GameObject nodeObj, ExplorationNode node)
        {
            var renderer = nodeObj.GetComponent<Renderer>();
            if (renderer == null) return;
            
            Color nodeColor = GetNodeTypeColor(node.type);
            renderer.material.color = nodeColor;
        }
        
        /// <summary>
        /// Get color for node type
        /// </summary>
        private Color GetNodeTypeColor(NodeType nodeType)
        {
            switch (nodeType)
            {
                case NodeType.Start: return startNodeColor;
                case NodeType.Combat: return combatNodeColor;
                case NodeType.Resource: return resourceNodeColor;
                case NodeType.Event: return eventNodeColor;
                case NodeType.Danger: return dangerNodeColor;
                case NodeType.Repair: return repairNodeColor;
                case NodeType.Extraction: return extractionNodeColor;
                case NodeType.Boss: return bossNodeColor;
                case NodeType.Shop: return shopNodeColor;
                case NodeType.Rest: return restNodeColor;
                default: return Color.white;
            }
        }
        
        /// <summary>
        /// Set node material based on state
        /// </summary>
        private void SetNodeMaterial(GameObject nodeObj, ExplorationNode node)
        {
            var renderer = nodeObj.GetComponent<Renderer>();
            if (renderer == null) return;
            
            Material stateMaterial = GetNodeStateMaterial(node.state);
            if (stateMaterial != null)
            {
                renderer.material = stateMaterial;
            }
        }
        
        /// <summary>
        /// Get material for node state
        /// </summary>
        private Material GetNodeStateMaterial(NodeState state)
        {
            switch (state)
            {
                case NodeState.Hidden: return hiddenNodeMaterial;
                case NodeState.Visible: return visibleNodeMaterial;
                case NodeState.Accessible: return accessibleNodeMaterial;
                case NodeState.Completed: return completedNodeMaterial;
                default: return null;
            }
        }
        
        /// <summary>
        /// Get material for connection
        /// </summary>
        private Material GetConnectionMaterial(ExplorationNode nodeA, ExplorationNode nodeB)
        {
            // Use accessible material if either node is accessible
            if (nodeA.state == NodeState.Accessible || nodeB.state == NodeState.Accessible)
            {
                return accessibleNodeMaterial;
            }
            
            // Use visible material if both nodes are visible
            if (nodeA.state != NodeState.Hidden && nodeB.state != NodeState.Hidden)
            {
                return visibleNodeMaterial;
            }
            
            return hiddenNodeMaterial;
        }
        
        /// <summary>
        /// Handle node discovery event
        /// </summary>
        private void OnNodeDiscovered(ExplorationNode node)
        {
            Debug.Log($"GraphVisualizer: Node discovered: {node.nodeId} at {node.position}");
            CreateNodeVisual(node);
            UpdateConnections();
        }
        
        /// <summary>
        /// Handle node completion event
        /// </summary>
        private void OnNodeCompleted(ExplorationNode node)
        {
            if (nodeObjects.ContainsKey(node.position))
            {
                SetNodeMaterial(nodeObjects[node.position], node);
            }
            UpdateConnections();
            
            // Also check for newly accessible nodes after completion
            UpdateVisibleNodes();
        }
        
        /// <summary>
        /// Update connection visuals
        /// </summary>
        private void UpdateConnections()
        {
            Debug.Log("GraphVisualizer: Updating connections...");
            
            // Clear old connections
            foreach (var connection in connectionObjects.Values)
            {
                if (connection != null)
                {
                    DestroyImmediate(connection);
                }
            }
            connectionObjects.Clear();
            
            // Recreate connections
            CreateConnections();
        }
        
        /// <summary>
        /// Force refresh all connections
        /// </summary>
        public void RefreshConnections()
        {
            Debug.Log("GraphVisualizer: Force refreshing all connections...");
            UpdateConnections();
        }
        
        /// <summary>
        /// Test method to create sample connections for debugging
        /// </summary>
        public void TestConnections()
        {
            Debug.Log("GraphVisualizer: Testing connection creation...");
            
            // Create test nodes at different positions
            var testNodeA = new ExplorationNode(new Vector2Int(0, 0), NodeType.Start);
            testNodeA.nodeId = "test_a";
            
            var testNodeB = new ExplorationNode(new Vector2Int(1, 0), NodeType.Resource); // Right
            testNodeB.nodeId = "test_b";
            
            var testNodeC = new ExplorationNode(new Vector2Int(0, 1), NodeType.Resource); // Up
            testNodeC.nodeId = "test_c";
            
            var testNodeD = new ExplorationNode(new Vector2Int(-1, 0), NodeType.Resource); // Left
            testNodeD.nodeId = "test_d";
            
            var testNodeE = new ExplorationNode(new Vector2Int(0, -1), NodeType.Resource); // Down
            testNodeE.nodeId = "test_e";
            
            // Test connections in all four directions
            Debug.Log("Testing horizontal connection (right)");
            CreateConnection(testNodeA, testNodeB);
            
            Debug.Log("Testing vertical connection (up)");
            CreateConnection(testNodeA, testNodeC);
            
            Debug.Log("Testing horizontal connection (left)");
            CreateConnection(testNodeA, testNodeD);
            
            Debug.Log("Testing vertical connection (down)");
            CreateConnection(testNodeA, testNodeE);
            
            Debug.Log($"Test complete. Created {connectionObjects.Count} test connections");
        }
        
        /// <summary>
        /// Clear test connections
        /// </summary>
        public void ClearTestConnections()
        {
            Debug.Log("GraphVisualizer: Clearing test connections...");
            
            // Find and remove test connections
            var testConnectionKeys = connectionObjects.Keys.Where(k => k.StartsWith("test_")).ToList();
            foreach (var key in testConnectionKeys)
            {
                if (connectionObjects[key] != null)
                {
                    DestroyImmediate(connectionObjects[key]);
                }
                connectionObjects.Remove(key);
            }
            
            Debug.Log($"Cleared {testConnectionKeys.Count} test connections");
        }
        
        /// <summary>
        /// Visualize current connection state
        /// </summary>
        public void VisualizeConnections()
        {
            Debug.Log("GraphVisualizer: Visualizing current connections...");
            
            var allNodes = graphSystem.GetVisibleNodes();
            foreach (var node in allNodes)
            {
                Debug.Log($"Node {node.nodeId} at {node.position} has {node.connectedNodeIds.Count} connections:");
                foreach (var connectedId in node.connectedNodeIds)
                {
                    var connectedNode = allNodes.FirstOrDefault(n => n.nodeId == connectedId);
                    if (connectedNode != null)
                    {
                        var distance = Vector2Int.Distance(node.position, connectedNode.position);
                        Debug.Log($"  -> {connectedId} at {connectedNode.position} (distance: {distance})");
                    }
                }
            }
        }
        
        /// <summary>
        /// Get all node objects
        /// </summary>
        public Dictionary<Vector2Int, GameObject> GetNodeObjects()
        {
            return nodeObjects;
        }
        
        /// <summary>
        /// Get node object at position
        /// </summary>
        public GameObject GetNodeObject(Vector2Int position)
        {
            return nodeObjects.ContainsKey(position) ? nodeObjects[position] : null;
        }
        
        /// <summary>
        /// Clear all visual elements
        /// </summary>
        public void ClearVisualization()
        {
            if (graphContainer != null)
            {
                DestroyImmediate(graphContainer.gameObject);
            }
            
            nodeObjects.Clear();
            connectionObjects.Clear();
            
            InitializeVisualizer();
        }
        
        /// <summary>
        /// Handle new chunk generation event
        /// </summary>
        private void OnNewChunkGenerated(int chunkId)
        {
            Debug.Log($"GraphVisualizer: New chunk {chunkId} generated, updating visualization...");
            
            // Get all newly generated nodes
            var allNodes = graphSystem.GetVisibleNodes();
            var newNodes = new List<ExplorationNode>();
            
            foreach (var node in allNodes)
            {
                if (!nodeObjects.ContainsKey(node.position))
                {
                    newNodes.Add(node);
                }
            }
            
            // Create visuals for new nodes
            foreach (var node in newNodes)
            {
                CreateNodeVisual(node);
            }
            
            // Update connections to include new nodes
            UpdateConnections();
            
            Debug.Log($"GraphVisualizer: Added {newNodes.Count} new nodes from chunk {chunkId}");
        }

        /// <summary>
        /// Update visualization to include newly visible nodes
        /// </summary>
        private void UpdateVisibleNodes()
        {
            if (graphSystem == null) return;
            
            var allVisibleNodes = graphSystem.GetVisibleNodes();
            var newNodes = new List<ExplorationNode>();
            
            foreach (var node in allVisibleNodes)
            {
                if (!nodeObjects.ContainsKey(node.position))
                {
                    newNodes.Add(node);
                }
            }
            
            if (newNodes.Count > 0)
            {
                Debug.Log($"GraphVisualizer: Found {newNodes.Count} new visible nodes, updating visualization...");
                
                foreach (var node in newNodes)
                {
                    CreateNodeVisual(node);
                }
                
                UpdateConnections();
            }
        }

        /// <summary>
        /// Reveal nodes around the player's current position
        /// </summary>
        public void RevealNodesAroundPlayer(string partyId)
        {
            if (graphSystem == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Debug.Log($"GraphVisualizer: Revealing nodes around party {partyId} at position {party.currentNodePosition}");
            
            // Get all visible nodes and check if we need to create visuals for any
            var allVisibleNodes = graphSystem.GetVisibleNodes();
            var nodesToReveal = new List<ExplorationNode>();
            
            foreach (var node in allVisibleNodes)
            {
                if (!nodeObjects.ContainsKey(node.position))
                {
                    nodesToReveal.Add(node);
                }
            }
            
            if (nodesToReveal.Count > 0)
            {
                Debug.Log($"GraphVisualizer: Revealing {nodesToReveal.Count} new nodes around player");
                
                foreach (var node in nodesToReveal)
                {
                    CreateNodeVisual(node);
                }
                
                UpdateConnections();
            }
        }

        /// <summary>
        /// Force update of all visible nodes
        /// </summary>
        public void ForceUpdateVisibleNodes()
        {
            Debug.Log("GraphVisualizer: Force updating all visible nodes...");
            UpdateVisibleNodes();
        }

        /// <summary>
        /// Log current graph state for debugging
        /// </summary>
        public void LogGraphState()
        {
            if (graphSystem == null)
            {
                Debug.Log("GraphVisualizer: No GraphSystem found!");
                return;
            }
            
            var allNodes = graphSystem.GetVisibleNodes();
            Debug.Log($"GraphVisualizer: Current graph state - {allNodes.Count} nodes, {connectionObjects.Count} visual connections");
            
            foreach (var node in allNodes)
            {
                Debug.Log($"  Node {node.nodeId}: type={node.type}, position={node.position}, state={node.state}, connections={node.connectedNodeIds.Count}");
                foreach (var connectedId in node.connectedNodeIds)
                {
                    Debug.Log($"    -> Connected to {connectedId}");
                }
            }
        }

        /// <summary>
        /// Public method to manually update the visualization
        /// Call this when you want to force a refresh of the graph visualization
        /// </summary>
        public void UpdateVisualization()
        {
            Debug.Log("GraphVisualizer: Manual visualization update requested...");
            UpdateVisibleNodes();
            UpdateConnections();
        }
    }
    
    /// <summary>
    /// Component attached to node objects to store node information
    /// </summary>
    public class NodeInfo : MonoBehaviour
    {
        public ExplorationNode node;
        
        public void Initialize(ExplorationNode nodeData)
        {
            node = nodeData;
        }
        
        void OnMouseDown()
        {
            // Handle node selection
            Debug.Log($"Selected node: {node.type} at {node.position}");
        }
        
        void OnMouseEnter()
        {
            // Show node tooltip
            transform.localScale *= 1.2f;
        }
        
        void OnMouseExit()
        {
            // Hide node tooltip
            transform.localScale /= 1.2f;
        }
    }
}

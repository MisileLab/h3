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
        private bool isInitialized = false;
        private GameObject graphSystemGameObject; // Track the GraphSystem GameObject
        
        void Start()
        {
            // Prevent this GameObject from being destroyed when scenes change
            DontDestroyOnLoad(gameObject);
            
            InitializeVisualizer();
            
            // Validate required components
            ValidateRequiredComponents();
            
            // Use a more robust approach to find GraphSystem
            // Wait a frame to ensure GraphSystem is initialized
            StartCoroutine(InitializeAfterGraphSystem());
        }
        
        void OnDestroy()
        {
            Debug.LogWarning("GraphVisualizer: GameObject is being destroyed! This should not happen with DontDestroyOnLoad.");
        }
        
        void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus)
            {
                Debug.Log("GraphVisualizer: Application paused");
            }
            else
            {
                Debug.Log("GraphVisualizer: Application resumed");
            }
        }
        
        /// <summary>
        /// Validate that required components are assigned
        /// </summary>
        private void ValidateRequiredComponents()
        {
            if (nodePrefab == null)
            {
                Debug.LogWarning("GraphVisualizer: No nodePrefab assigned! Will use default spheres.");
            }
            
            if (connectionPrefab == null)
            {
                Debug.LogWarning("GraphVisualizer: No connectionPrefab assigned! Will use LineRenderer.");
            }
            
            if (hiddenNodeMaterial == null)
            {
                Debug.LogWarning("GraphVisualizer: No hiddenNodeMaterial assigned!");
            }
            
            if (visibleNodeMaterial == null)
            {
                Debug.LogWarning("GraphVisualizer: No visibleNodeMaterial assigned!");
            }
            
            if (accessibleNodeMaterial == null)
            {
                Debug.LogWarning("GraphVisualizer: No accessibleNodeMaterial assigned!");
            }
            
            if (completedNodeMaterial == null)
            {
                Debug.LogWarning("GraphVisualizer: No completedNodeMaterial assigned!");
            }
        }
        
        /// <summary>
        /// Coroutine to initialize after GraphSystem is ready
        /// </summary>
        private System.Collections.IEnumerator InitializeAfterGraphSystem()
        {
            // Wait a few frames to ensure GraphSystem is initialized
            yield return null;
            yield return null;
            
            // Try to find GraphSystem
            graphSystem = FindFirstObjectByType<GraphSystem>();
            if (graphSystem == null)
            {
                Debug.LogError("GraphVisualizer: No GraphSystem found in scene! Retrying...");
                
                // Wait a bit more and try again
                yield return new WaitForSeconds(0.1f);
                graphSystem = FindFirstObjectByType<GraphSystem>();
                
                if (graphSystem == null)
                {
                    Debug.LogError("GraphVisualizer: Still no GraphSystem found after retry!");
                    yield break;
                }
            }
            
            // Store reference to the GraphSystem GameObject
            graphSystemGameObject = graphSystem.gameObject;
            Debug.Log($"GraphVisualizer: Found GraphSystem on GameObject: {graphSystemGameObject.name}");
            
            // Check if the GraphSystem GameObject is active and enabled
            if (!graphSystemGameObject.activeInHierarchy)
            {
                Debug.LogError("GraphVisualizer: GraphSystem GameObject is not active in hierarchy!");
                yield break;
            }
            
            if (!graphSystem.enabled)
            {
                Debug.LogError("GraphVisualizer: GraphSystem component is disabled!");
                yield break;
            }
            
            // Wait for GraphSystem to be fully initialized
            while (!graphSystem.IsGraphGenerated())
            {
                Debug.Log("GraphVisualizer: Waiting for GraphSystem to generate graph...");
                yield return new WaitForSeconds(0.1f);
                
                // Safety check - don't wait forever
                if (Time.time > 10f)
                {
                    Debug.LogError("GraphVisualizer: Timeout waiting for GraphSystem to generate graph!");
                    yield break;
                }
            }
            
            // Additional check - verify the graph actually has nodes
            var nodeCount = graphSystem.GetTotalNodeCount();
            if (nodeCount == 0)
            {
                Debug.LogError("GraphVisualizer: GraphSystem reports 0 nodes even though IsGraphGenerated() is true!");
                yield break;
            }
            
            Debug.Log($"GraphVisualizer: GraphSystem is ready with {nodeCount} nodes!");
            
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
            
            Debug.Log("GraphVisualizer: Initialization complete!");
            isInitialized = true; // Set flag after successful initialization
        }
        
        void Update()
        {
            // If we lost the GraphSystem reference, try to find it again
            if (isInitialized && graphSystem == null)
            {
                Debug.LogWarning("GraphVisualizer: Lost GraphSystem reference, attempting to reconnect...");
                isInitialized = false;
                StartCoroutine(InitializeAfterGraphSystem());
                return;
            }
            
            // Check if the GraphSystem GameObject is being destroyed
            if (isInitialized && graphSystemGameObject == null)
            {
                Debug.LogError("GraphVisualizer: GraphSystem GameObject was destroyed! This should not happen.");
                isInitialized = false;
                graphSystem = null;
                return;
            }
            
            // Check if the GraphSystem component is still valid
            if (isInitialized && graphSystem != null && graphSystem.Equals(null))
            {
                Debug.LogError("GraphVisualizer: GraphSystem component was destroyed! This should not happen.");
                isInitialized = false;
                graphSystem = null;
                graphSystemGameObject = null;
                return;
            }
            
            // Check if visualizer is properly initialized before proceeding
            if (!isInitialized || graphSystem == null) 
            {
                // Add debug info to help identify initialization issues
                if (Time.frameCount % 60 == 0) // Log every 60 frames (about once per second)
                {
                    Debug.Log($"GraphVisualizer: Waiting for initialization... isInitialized={isInitialized}, graphSystem={graphSystem != null}, graphSystemGameObject={graphSystemGameObject != null}");
                }
                return;
            }
            
            // Update graph visualization when it's ready
            // Note: Camera movement conflicts have been addressed by:
            // 1. CameraZoomSystem now waits 1 second before focusing on main party
            // 2. PartyVisualizer auto camera movement is disabled by default
            if (graphSystem.IsGraphGenerated())
            {
                if (nodeObjects.Count == 0)
                {
                    Debug.Log("GraphVisualizer: Graph is generated, creating visualization...");
                    CreateGraphVisualization();
                }
                else
                {
                    // Continuously check for newly visible nodes
                    UpdateVisibleNodes();
                }
            }
            else
            {
                // Add debug info to track graph generation status
                if (Time.frameCount % 120 == 0) // Log every 120 frames (about every 2 seconds)
                {
                    Debug.Log($"GraphVisualizer: Waiting for graph generation... IsGraphGenerated={graphSystem.IsGraphGenerated()}");
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
            if (graphSystem == null) 
            {
                Debug.LogError("GraphVisualizer: Cannot create visualization - GraphSystem is null!");
                return;
            }
            
            if (!graphSystem.IsGraphGenerated())
            {
                Debug.LogWarning("GraphVisualizer: Cannot create visualization - Graph is not yet generated!");
                return;
            }
            
            Debug.Log("GraphVisualizer: Starting to create graph visualization...");
            
            // Create all nodes - including branch nodes
            var allNodes = graphSystem.GetVisibleNodes();
            Debug.Log($"GraphVisualizer: Found {allNodes.Count} visible nodes to visualize");
            
            // Log details about each node to help debug branch node issues
            foreach (var node in allNodes)
            {
                Debug.Log($"GraphVisualizer: Node {node.nodeId}: type={node.type}, position={node.position}, state={node.state}, connections={node.connectedNodeIds.Count}");
            }
            
            // Separate branch nodes for special handling
            var branchNodes = allNodes.Where(n => n.position.y > 2 || n.position.y < 0).ToList();
            var pathNodes = allNodes.Where(n => n.position.y <= 2 && n.position.y >= 0).ToList();
            
            Debug.Log($"GraphVisualizer: Found {pathNodes.Count} path nodes and {branchNodes.Count} branch nodes");
            
            // Create visuals for all nodes
            foreach (var node in allNodes)
            {
                CreateNodeVisual(node);
            }
            
            // Validate connections before creating visuals
            ValidateConnections();
            
            // Create connections
            CreateConnections();
            
            Debug.Log($"GraphVisualizer: Created visualization with {nodeObjects.Count} nodes and {connectionObjects.Count} connections");
            
            // Verify that all nodes have visuals
            var missingNodes = allNodes.Where(n => !nodeObjects.ContainsKey(n.position)).ToList();
            if (missingNodes.Count > 0)
            {
                Debug.LogWarning($"GraphVisualizer: {missingNodes.Count} nodes are missing visuals after creation!");
                foreach (var missingNode in missingNodes)
                {
                    Debug.LogWarning($"  Missing visual for node {missingNode.nodeId} at {missingNode.position}");
                }
            }
            else
            {
                Debug.Log("GraphVisualizer: All nodes have visuals created successfully");
            }
        }
        
        /// <summary>
        /// Force refresh the entire visualization
        /// Call this if the visualization seems broken or incomplete
        /// </summary>
        public void ForceRefreshVisualization()
        {
            Debug.Log("GraphVisualizer: Force refreshing entire visualization...");
            
            // Clear existing visualization
            ClearVisualization();
            
            // Wait a frame and recreate
            StartCoroutine(RefreshAfterDelay());
        }
        
        private System.Collections.IEnumerator RefreshAfterDelay()
        {
            yield return null;
            
            if (graphSystem != null && graphSystem.IsGraphGenerated())
            {
                CreateGraphVisualization();
            }
            else
            {
                Debug.LogWarning("GraphVisualizer: Cannot refresh - GraphSystem not ready or graph not generated");
            }
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
            if (graphSystem == null) return;
            
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
            if (graphSystem == null) return;
            
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
            Debug.Log($"GraphVisualizer: Node discovered: {node.nodeId} at {node.position} (type: {node.type}, state: {node.state})");
            
            // Immediately create visual for the discovered node
            if (!nodeObjects.ContainsKey(node.position))
            {
                CreateNodeVisual(node);
                Debug.Log($"GraphVisualizer: Created visual for newly discovered node {node.nodeId}");
            }
            else
            {
                Debug.Log($"GraphVisualizer: Node {node.nodeId} already has a visual object");
            }
            
            // Update connections to include the new node
            UpdateConnections();
            
            // Force update of all visible nodes to catch any branch nodes that might have been revealed
            UpdateVisibleNodes();
            
            // Also check if this node reveals any connected nodes that should be visualized
            CheckAndRevealConnectedNodes(node);
        }

        /// <summary>
        /// Check if a discovered node reveals any connected nodes that should be visualized
        /// </summary>
        private void CheckAndRevealConnectedNodes(ExplorationNode discoveredNode)
        {
            if (graphSystem == null) return;
            
            Debug.Log($"GraphVisualizer: Checking connected nodes for {discoveredNode.nodeId} at {discoveredNode.position}");
            
            var allVisibleNodes = graphSystem.GetVisibleNodes();
            var connectedNodes = new List<ExplorationNode>();
            
            // Find all nodes that are connected to the discovered node
            foreach (var connectedId in discoveredNode.connectedNodeIds)
            {
                var connectedNode = allVisibleNodes.FirstOrDefault(n => n.nodeId == connectedId);
                if (connectedNode != null)
                {
                    connectedNodes.Add(connectedNode);
                    Debug.Log($"GraphVisualizer: Found connected node {connectedNode.nodeId} at {connectedNode.position} (type: {connectedNode.type}, state: {connectedNode.state})");
                }
            }
            
            // Create visuals for any connected nodes that don't have them
            foreach (var connectedNode in connectedNodes)
            {
                if (!nodeObjects.ContainsKey(connectedNode.position))
                {
                    Debug.Log($"GraphVisualizer: Creating visual for connected node {connectedNode.nodeId} at {connectedNode.position}");
                    CreateNodeVisual(connectedNode);
                }
            }
            
            // Update connections if we added new nodes
            if (connectedNodes.Any(n => !nodeObjects.ContainsKey(n.position)))
            {
                UpdateConnections();
            }
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
            if (graphSystem == null) return;
            
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
            if (graphSystem == null) return;
            
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
        /// Force refresh branch nodes specifically
        /// Call this when you suspect branch nodes are not being shown
        /// </summary>
        public void ForceRefreshBranchNodes()
        {
            Debug.Log("GraphVisualizer: Force refreshing branch nodes...");
            
            if (graphSystem == null) return;
            
            var allNodes = graphSystem.GetVisibleNodes();
            var branchNodes = allNodes.Where(n => n.position.y > 2 || n.position.y < 0).ToList();
            
            Debug.Log($"GraphVisualizer: Found {branchNodes.Count} branch nodes to refresh");
            
            foreach (var branchNode in branchNodes)
            {
                if (!nodeObjects.ContainsKey(branchNode.position))
                {
                    Debug.Log($"GraphVisualizer: Creating missing visual for branch node {branchNode.nodeId} at {branchNode.position}");
                    CreateNodeVisual(branchNode);
                }
                else
                {
                    Debug.Log($"GraphVisualizer: Branch node {branchNode.nodeId} already has visual at {branchNode.position}");
                }
            }
            
            // Update connections to include any new branch nodes
            UpdateConnections();
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
            
            // First, check for any missing branch nodes
            ForceRefreshBranchNodes();
            
            // Then update all visible nodes
            UpdateVisibleNodes();
            
            // Finally, update connections
            UpdateConnections();
        }
        
        /// <summary>
        /// Public method to manually trigger initialization
        /// Call this if the visualizer needs to be reinitialized
        /// </summary>
        public void ManualInitialize()
        {
            if (isInitialized)
            {
                Debug.Log("GraphVisualizer: Already initialized, skipping manual initialization");
                return;
            }
            
            Debug.Log("GraphVisualizer: Manual initialization requested...");
            StartCoroutine(InitializeAfterGraphSystem());
        }
        
        /// <summary>
        /// Check if the visualizer is properly initialized
        /// </summary>
        public bool IsInitialized()
        {
            return isInitialized && graphSystem != null;
        }
        
        /// <summary>
        /// Get the current GraphSystem reference
        /// </summary>
        public GraphSystem GetGraphSystem()
        {
            return graphSystem;
        }

        /// <summary>
        /// Comprehensive debug method to diagnose visualizer issues
        /// </summary>
        public void DebugVisualizerState()
        {
            Debug.Log("=== GraphVisualizer Debug State ===");
            Debug.Log($"isInitialized: {isInitialized}");
            Debug.Log($"graphSystem: {(graphSystem != null ? "Found" : "NULL")}");
            Debug.Log($"graphSystemGameObject: {(graphSystemGameObject != null ? "Found" : "NULL")}");
            
            if (graphSystem != null)
            {
                Debug.Log($"IsGraphGenerated: {graphSystem.IsGraphGenerated()}");
                Debug.Log($"Total nodes in system: {graphSystem.GetTotalNodeCount()}");
                Debug.Log($"Visible nodes in system: {graphSystem.GetVisibleNodes().Count}");
                
                var visibleNodes = graphSystem.GetVisibleNodes();
                Debug.Log("Visible nodes details:");
                foreach (var node in visibleNodes.Take(5)) // Show first 5 nodes
                {
                    Debug.Log($"  Node {node.nodeId}: type={node.type}, position={node.position}, state={node.state}");
                }
                if (visibleNodes.Count > 5)
                {
                    Debug.Log($"  ... and {visibleNodes.Count - 5} more nodes");
                }
                
                // Check for branch nodes specifically
                var branchNodes = visibleNodes.Where(n => n.position.y > 2 || n.position.y < 0).ToList();
                Debug.Log($"Branch nodes: {branchNodes.Count} out of {visibleNodes.Count} visible nodes");
                
                foreach (var branchNode in branchNodes)
                {
                    var hasVisual = nodeObjects.ContainsKey(branchNode.position);
                    Debug.Log($"  Branch node {branchNode.nodeId}: type={branchNode.type}, position={branchNode.position}, state={branchNode.state}, hasVisual={hasVisual}");
                }
            }
            
            if (graphSystemGameObject != null)
            {
                Debug.Log($"GraphSystem GameObject active: {graphSystemGameObject.activeInHierarchy}");
                Debug.Log($"GraphSystem GameObject name: {graphSystemGameObject.name}");
                Debug.Log($"GraphSystem GameObject position: {graphSystemGameObject.transform.position}");
            }
            
            Debug.Log($"nodeObjects count: {nodeObjects.Count}");
            Debug.Log($"connectionObjects count: {connectionObjects.Count}");
            Debug.Log($"graphContainer: {(graphContainer != null ? "Exists" : "NULL")}");
            
            if (graphContainer != null)
            {
                Debug.Log($"graphContainer child count: {graphContainer.childCount}");
                Debug.Log($"graphContainer position: {graphContainer.position}");
            }
            
            Debug.Log($"Current frame: {Time.frameCount}");
            Debug.Log($"Current time: {Time.time}");
            Debug.Log("=== End Debug State ===");
        }

        /// <summary>
        /// Check for visualization mismatches and fix them
        /// </summary>
        public void DiagnoseAndFixVisualization()
        {
            Debug.Log("GraphVisualizer: Diagnosing visualization issues...");
            
            if (graphSystem == null)
            {
                Debug.LogError("GraphVisualizer: Cannot diagnose - GraphSystem is null");
                return;
            }
            
            var visibleNodes = graphSystem.GetVisibleNodes();
            var missingNodes = new List<ExplorationNode>();
            
            foreach (var node in visibleNodes)
            {
                if (!nodeObjects.ContainsKey(node.position))
                {
                    missingNodes.Add(node);
                    Debug.Log($"GraphVisualizer: Missing visual for node {node.nodeId} at {node.position}");
                }
            }
            
            if (missingNodes.Count > 0)
            {
                Debug.Log($"GraphVisualizer: Found {missingNodes.Count} missing node visuals, creating them...");
                
                foreach (var node in missingNodes)
                {
                    CreateNodeVisual(node);
                }
                
                UpdateConnections();
                Debug.Log("GraphVisualizer: Fixed missing node visuals");
            }
            else
            {
                Debug.Log("GraphVisualizer: No missing node visuals found");
            }
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

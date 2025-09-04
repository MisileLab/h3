using UnityEngine;
using System.Collections.Generic;
using UnityEngine.InputSystem;

namespace Scalar
{
    /// <summary>
    /// Test script to demonstrate the GraphSystem functionality
    /// </summary>
    public class GraphSystemTest : MonoBehaviour
    {
        // Input System variables
        private InputAction completeNodeAction;
        private InputAction regenerateAction;
        private InputAction clickAction;
        private InputAction generateChunkAction;
        
        [Header("Debug Info")]
        [SerializeField] private bool showDebugInfo = true;
        [SerializeField] private bool logNodeEvents = true;
        
        private GraphSystem graphSystem;
        private GraphVisualizer graphVisualizer;
        private CameraZoomSystem cameraZoomSystem;
        private Camera mainCamera;
        
        void Start()
        {
            // Find required components
            graphSystem = FindFirstObjectByType<GraphSystem>();
            graphVisualizer = FindFirstObjectByType<GraphVisualizer>();
            cameraZoomSystem = FindFirstObjectByType<CameraZoomSystem>();
            mainCamera = Camera.main;
            
            if (graphSystem == null)
            {
                Debug.LogError("GraphSystemTest: No GraphSystem found in scene!");
                return;
            }
            
            // Initialize Input System
            InitializeInputSystem();
            
            // Subscribe to events for logging
            if (logNodeEvents)
            {
                graphSystem.OnNodeDiscovered.AddListener(OnNodeDiscovered);
                graphSystem.OnNodeCompleted.AddListener(OnNodeCompleted);
                graphSystem.OnPartyDestroyed.AddListener(OnPartyDestroyed);
                graphSystem.OnMachineDestroyed.AddListener(OnMachineDestroyed);
                graphSystem.OnMachineRepaired.AddListener(OnMachineRepaired);
            }
            
            Debug.Log("GraphSystemTest: Test system initialized. Use WASD to move parties, Space to complete nodes, R to regenerate graph, G to generate next chunk.");
        }
        
        void Update()
        {
            if (graphSystem == null) return;
            
            UpdateDebugInfo();
        }
        
        /// <summary>
        /// Initialize the Input System
        /// </summary>
        private void InitializeInputSystem()
        {
            // Create individual input actions for each key
            var wAction = new InputAction("W", InputActionType.Button, "<Keyboard>/w");
            var aAction = new InputAction("A", InputActionType.Button, "<Keyboard>/a");
            var sAction = new InputAction("S", InputActionType.Button, "<Keyboard>/s");
            var dAction = new InputAction("D", InputActionType.Button, "<Keyboard>/d");
            
            completeNodeAction = new InputAction("CompleteNode", InputActionType.Button, "<Keyboard>/space");
            regenerateAction = new InputAction("Regenerate", InputActionType.Button, "<Keyboard>/r");
            clickAction = new InputAction("Click", InputActionType.Button, "<Mouse>/leftButton");
            generateChunkAction = new InputAction("GenerateChunk", InputActionType.Button, "<Keyboard>/g");
            
            // Subscribe to input events
            wAction.performed += ctx => MoveParty("party_main", new Vector2Int(0, 1));
            aAction.performed += ctx => MoveParty("party_main", new Vector2Int(-1, 0));
            sAction.performed += ctx => MoveParty("party_main", new Vector2Int(0, -1));
                        dAction.performed += ctx => MoveParty("party_main", new Vector2Int(1, 0));
            
            completeNodeAction.performed += OnCompleteNodePerformed;
            regenerateAction.performed += OnRegeneratePerformed;
            clickAction.performed += OnClickPerformed;
            generateChunkAction.performed += OnGenerateChunkPerformed;
            
            // Add key binding for focusing on main party (F key)
            var focusPartyAction = new InputAction("FocusParty", InputActionType.Button, "<Keyboard>/f");
            focusPartyAction.performed += ctx => FocusOnMainParty();
            focusPartyAction.Enable();
            
            // Enable the actions
            wAction.Enable();
            aAction.Enable();
            sAction.Enable();
            dAction.Enable();
            completeNodeAction.Enable();
            regenerateAction.Enable();
            clickAction.Enable();
            generateChunkAction.Enable();
            
            Debug.Log("Input System initialized for GraphSystemTest");
        }
        

        

        
        private void OnCompleteNodePerformed(InputAction.CallbackContext context)
        {
            CompleteCurrentNode();
        }
        
        private void OnRegeneratePerformed(InputAction.CallbackContext context)
        {
            RegenerateGraph();
        }
        
        private void OnGenerateChunkPerformed(InputAction.CallbackContext context)
        {
            GenerateNextChunk();
        }
        
        private void OnClickPerformed(InputAction.CallbackContext context)
        {
            HandleMouseClick();
        }
        

        
        /// <summary>
        /// Move a party in the specified direction
        /// </summary>
        private void MoveParty(string partyId, Vector2Int direction)
        {
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector2Int targetPosition = party.currentNodePosition + direction;
            
            if (graphSystem.MoveParty(partyId, targetPosition))
            {
                Debug.Log($"Moved {party.name} to {targetPosition}");
                
                // Update the visualization to show newly discovered nodes
                if (graphVisualizer != null)
                {
                    graphVisualizer.UpdateVisualization();
                }
            }
            else
            {
                Debug.Log($"Cannot move {party.name} to {targetPosition}");
            }
        }
        
        /// <summary>
        /// Complete the current node for the main party
        /// </summary>
        private void CompleteCurrentNode()
        {
            var mainParty = graphSystem.GetParty("party_main");
            if (mainParty == null) return;
            
            var currentNode = graphSystem.GetNodeAt(mainParty.currentNodePosition);
            if (currentNode != null && currentNode.state == NodeState.Accessible)
            {
                graphSystem.CompleteNode(mainParty.currentNodePosition);
                Debug.Log($"Completed {currentNode.type} node at {currentNode.position}");
            }
            else
            {
                Debug.Log("No accessible node to complete");
            }
        }
        
        /// <summary>
        /// Regenerate the entire graph
        /// </summary>
        private void RegenerateGraph()
        {
            Debug.Log("Regenerating graph...");
            
            // Clear visualization
            if (graphVisualizer != null)
            {
                graphVisualizer.ClearVisualization();
            }
            
            // Regenerate the graph using the GraphSystem
            if (graphSystem != null)
            {
                graphSystem.RegenerateGraph();
                Debug.Log("Graph regenerated successfully");
            }
            else
            {
                Debug.LogError("Cannot regenerate graph - GraphSystem not found");
            }
        }
        
        /// <summary>
        /// Manually generate the next chunk
        /// </summary>
        private void GenerateNextChunk()
        {
            if (graphSystem == null) return;

            graphSystem.GenerateNextChunk();
            Debug.Log("Manually generated next chunk.");
        }

        /// <summary>
        /// Manually update the graph visualization
        /// </summary>
        public void UpdateVisualization()
        {
            if (graphVisualizer != null)
            {
                graphVisualizer.UpdateVisualization();
                Debug.Log("Manually updated graph visualization");
            }
        }
        
        
        
        /// <summary>
        /// Handle mouse clicks for node selection
        /// </summary>
        private void HandleMouseClick()
        {
            Ray ray = mainCamera.ScreenPointToRay(Mouse.current.position.ReadValue());
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit))
            {
                var nodeInfo = hit.collider.GetComponent<NodeInfo>();
                if (nodeInfo != null)
                {
                    Debug.Log($"Clicked on {nodeInfo.node.type} node at {nodeInfo.node.position} (State: {nodeInfo.node.state})");
                    
                    // Show available actions for this node
                    ShowNodeActions(nodeInfo.node);
                }
            }
        }
        
        /// <summary>
        /// Show available actions for a node
        /// </summary>
        private void ShowNodeActions(ExplorationNode node)
        {
            Debug.Log($"--- Node Actions for {node.type} at {node.position} ---");
            
            switch (node.state)
            {
                case NodeState.Accessible:
                    Debug.Log("Actions: Complete node, Move party here");
                    break;
                case NodeState.Visible:
                    Debug.Log("Actions: None (not yet accessible)");
                    break;
                case NodeState.Completed:
                    Debug.Log("Actions: None (already completed)");
                    break;
                case NodeState.Hidden:
                    Debug.Log("Actions: None (hidden)");
                    break;
            }
            
            // Show connected nodes
            Debug.Log($"Connected to {node.connectedNodeIds.Count} nodes");
        }
        
        /// <summary>
        /// Update debug information display
        /// </summary>
        private void UpdateDebugInfo()
        {
            if (!showDebugInfo) return;
            
            // This would typically update a UI display
            // For now, we'll just log periodically
            if (Time.frameCount % 300 == 0) // Every 5 seconds at 60fps
            {
                LogGraphStatus();
            }
        }
        
        /// <summary>
        /// Log current graph status
        /// </summary>
        private void LogGraphStatus()
        {
            if (graphSystem == null) return;
            
            Debug.Log($"=== Graph Status ===");
            Debug.Log($"Total Nodes: {graphSystem.GetTotalNodeCount()}");
            Debug.Log($"Completed Nodes: {graphSystem.GetCompletedNodeCount()}");
            Debug.Log($"Current Depth: {graphSystem.GetCurrentDepth()}");
            Debug.Log($"Graph Generated: {graphSystem.IsGraphGenerated()}");
            
            // Show incremental generation info
            Debug.Log($"Incremental Generation: {graphSystem.GetLastGeneratedDepth()}/{graphSystem.GetMaxGeneratedDepth()} (Chunk Size: {graphSystem.GetChunkSize()})");
            Debug.Log($"Generated Chunks: {(graphSystem.GetLastGeneratedDepth() / graphSystem.GetChunkSize())}");
            
            // Show branch information
            var allNodes = graphSystem.GetAllNodes();
            int branchCount = 0;
            int pathCount = 0;
            foreach (var node in allNodes)
            {
                if (node.position.x > 0) // Not the start node
                {
                    if (node.position.y > 2 || node.position.y < 0) // Likely a branch
                    {
                        branchCount++;
                    }
                    else
                    {
                        pathCount++;
                    }
                }
            }
            Debug.Log($"Path Nodes: {pathCount}, Branch Nodes: {branchCount}");
            
            var parties = graphSystem.GetAllParties();
            foreach (var party in parties)
            {
                var currentNode = graphSystem.GetNodeAt(party.currentNodePosition);
                string nodeType = currentNode != null ? currentNode.type.ToString() : "Unknown";
                Debug.Log($"Party {party.name}: At {party.currentNodePosition} ({nodeType}), Energy: {party.energy}/{party.maxEnergy}");
            }
        }
        
        // Event handlers for logging
        private void OnNodeDiscovered(ExplorationNode node)
        {
            if (logNodeEvents)
            {
                Debug.Log($"🔍 DISCOVERED: {node.type} node at {node.position}");
            }
        }
        
        private void OnNodeCompleted(ExplorationNode node)
        {
            if (logNodeEvents)
            {
                Debug.Log($"✅ COMPLETED: {node.type} node at {node.position}");
            }
        }
        
        private void OnPartyDestroyed(Party party)
        {
            if (logNodeEvents)
            {
                Debug.Log($"💥 PARTY DESTROYED: {party.name}");
            }
        }
        
        private void OnMachineDestroyed(Machine machine)
        {
            if (logNodeEvents)
            {
                Debug.Log($"💀 MACHINE DESTROYED: {machine.name}");
            }
        }
        
        private void OnMachineRepaired(Machine machine)
        {
            if (logNodeEvents)
            {
                Debug.Log($"🔧 MACHINE REPAIRED: {machine.name}");
            }
        }
        
        /// <summary>
        /// Get debug information as a string
        /// </summary>
        public string GetDebugInfo()
        {
            if (graphSystem == null) return "GraphSystem not found";
            
            var parties = graphSystem.GetAllParties();
            string info = $"Graph Status:\n";
            info += $"Total Nodes: {graphSystem.GetTotalNodeCount()}\n";
            info += $"Completed: {graphSystem.GetCompletedNodeCount()}\n";
            info += $"Depth: {graphSystem.GetCurrentDepth()}\n";
            
            // Add incremental generation info
            info += $"Incremental Generation: {graphSystem.GetLastGeneratedDepth()}/{graphSystem.GetMaxGeneratedDepth()}\n";
            info += $"Generated Chunks: {(graphSystem.GetLastGeneratedDepth() / graphSystem.GetChunkSize())}\n";
            
            // Add branch information
            var allNodes = graphSystem.GetAllNodes();
            int branchCount = 0;
            int pathCount = 0;
            foreach (var node in allNodes)
            {
                if (node.position.x > 0) // Not the start node
                {
                    if (node.position.y > 2 || node.position.y < 0) // Likely a branch
                    {
                        branchCount++;
                    }
                    else
                    {
                        pathCount++;
                    }
                }
            }
            info += $"Path Nodes: {pathCount}, Branch Nodes: {branchCount}\n";
            
            info += "\n";
            
            info += "Parties:\n";
            foreach (var party in parties)
            {
                var currentNode = graphSystem.GetNodeAt(party.currentNodePosition);
                string nodeType = currentNode != null ? currentNode.type.ToString() : "Unknown";
                info += $"- {party.name}: {party.currentNodePosition} ({nodeType}), Energy: {party.energy}/{party.maxEnergy}\n";
            }
            
            return info;
        }
        
        /// <summary>
        /// Test function to simulate a combat encounter
        /// </summary>
        public void SimulateCombat(string partyId)
        {
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            var currentNode = graphSystem.GetNodeAt(party.currentNodePosition);
            if (currentNode == null || currentNode.type != NodeType.Combat) return;
            
            Debug.Log($"⚔️ Simulating combat for {party.name} at {currentNode.position}");
            
            // Simulate some damage to party members
            foreach (var machine in party.members)
            {
                int damage = Random.Range(10, 30);
                machine.currentHealth = Mathf.Max(0, machine.currentHealth - damage);
                
                if (machine.currentHealth <= 0)
                {
                    machine.isDestroyed = true;
                    OnMachineDestroyed(machine);
                }
                
                Debug.Log($"{machine.name} took {damage} damage. HP: {machine.currentHealth}/{machine.maxHealth}");
            }
            
            // Complete the combat node
            graphSystem.CompleteNode(party.currentNodePosition);
        }
        
        /// <summary>
        /// Test function to simulate resource collection
        /// </summary>
        public void SimulateResourceCollection(string partyId)
        {
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            var currentNode = graphSystem.GetNodeAt(party.currentNodePosition);
            if (currentNode == null || currentNode.type != NodeType.Resource) return;
            
            Debug.Log($"💎 Simulating resource collection for {party.name} at {currentNode.position}");
            
            // Simulate resource gain
            int energyGain = Random.Range(20, 50);
            party.energy = Mathf.Min(party.maxEnergy, party.energy + energyGain);
            
            Debug.Log($"{party.name} gained {energyGain} energy. Total: {party.energy}/{party.maxEnergy}");
            
            // Complete the resource node
            graphSystem.CompleteNode(party.currentNodePosition);
        }
        
        /// <summary>
        /// Focus camera on main party
        /// </summary>
        public void FocusOnMainParty()
        {
            var cameraSystem = FindFirstObjectByType<CameraZoomSystem>();
            if (cameraSystem != null)
            {
                cameraSystem.FocusOnMainPartyPublic();
                Debug.Log("GraphSystemTest: Focused camera on main party");
            }
            else
            {
                Debug.LogWarning("GraphSystemTest: No CameraZoomSystem found, cannot focus on main party");
            }
        }
        
        void OnDestroy()
        {
            // Clean up Input System actions
            if (completeNodeAction != null)
            {
                completeNodeAction.Disable();
                completeNodeAction.Dispose();
            }
            if (regenerateAction != null)
            {
                regenerateAction.Disable();
                regenerateAction.Dispose();
            }
            if (clickAction != null)
            {
                clickAction.Disable();
                clickAction.Dispose();
            }
            if (generateChunkAction != null)
            {
                generateChunkAction.Disable();
                generateChunkAction.Dispose();
            }
        }
    }
}

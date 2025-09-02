using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Events;

namespace Scalar
{
    /// <summary>
    /// Represents different types of nodes in the exploration graph
    /// </summary>
    public enum NodeType
    {
        Start,          // Starting point
        Combat,         // Turn-based combat encounters
        Resource,       // Collect parts, data, energy
        Event,          // Random events with choices
        Danger,         // High-risk, high-reward areas
        Repair,         // Repair destroyed machines
        Extraction,     // Safe extraction point
        Boss,           // Special boss encounters
        Shop,           // Trade resources
        Rest            // Heal and recover
    }

    /// <summary>
    /// Represents the state of a node
    /// </summary>
    public enum NodeState
    {
        Hidden,         // Not yet discovered (fog of war)
        Visible,        // Can see but not accessible
        Accessible,     // Can be reached by a party
        Completed,      // Node event finished
        Blocked,        // Temporarily inaccessible
        Destroyed       // Node is destroyed/damaged
    }

    /// <summary>
    /// Represents a party of machines
    /// </summary>
    [System.Serializable]
    public class Party
    {
        public string id;
        public string name;
        public List<Machine> members;
        public Vector2Int currentNodePosition;
        public bool isActive;
        public int energy;
        public int maxEnergy;
        
        public Party(string partyId, string partyName)
        {
            id = partyId;
            name = partyName;
            members = new List<Machine>();
            currentNodePosition = Vector2Int.zero;
            isActive = true;
            energy = 100;
            maxEnergy = 100;
        }
    }

    /// <summary>
    /// Represents a machine character
    /// </summary>
    [System.Serializable]
    public class Machine
    {
        public string id;
        public string name;
        public int maxHealth;
        public int currentHealth;
        public int attack;
        public int defense;
        public List<string> abilities;
        public bool isDestroyed;
        public Vector2Int lastKnownPosition;
        
        public Machine(string machineId, string machineName, int health, int atk, int def)
        {
            id = machineId;
            name = machineName;
            maxHealth = health;
            currentHealth = health;
            attack = atk;
            defense = def;
            abilities = new List<string>();
            isDestroyed = false;
            lastKnownPosition = Vector2Int.zero;
        }
    }

    /// <summary>
    /// Represents a node in the exploration graph
    /// </summary>
    [System.Serializable]
    public class ExplorationNode
    {
        public Vector2Int position;
        public NodeType type;
        public NodeState state;
        public string nodeId;
        public bool isVisited;
        public List<string> connectedNodeIds;
        public Dictionary<string, object> nodeData;
        public int difficulty;
        public int rewardTier;
        
        public ExplorationNode(Vector2Int pos, NodeType nodeType)
        {
            position = pos;
            type = nodeType;
            state = NodeState.Hidden;
            nodeId = System.Guid.NewGuid().ToString();
            isVisited = false;
            connectedNodeIds = new List<string>();
            nodeData = new Dictionary<string, object>();
            difficulty = 1;
            rewardTier = 1;
        }
    }

    /// <summary>
    /// Main graph system managing the exploration gameplay
    /// </summary>
    public class GraphSystem : MonoBehaviour
    {
        [Header("Graph Settings")]
        [SerializeField] private int gridWidth = 20;
        [SerializeField] private int gridHeight = 10;
        [SerializeField] private int maxDepth = 20; // Reduced to fit within gridWidth
        
        [Header("Generation Settings")]
        [SerializeField] private int minPathsPerRow = 2;
        [SerializeField] private int maxPathsPerRow = 3; // Reduced to ensure it fits within gridHeight
        [SerializeField] private float bossNodeFrequency = 0.1f;
        [SerializeField] private float shopNodeFrequency = 0.15f;
        
        [Header("Incremental Generation Settings")]
        [SerializeField] private int chunkSize = 10; // How many depth levels to generate at once
        [SerializeField] private int generationThreshold = 5; // Generate new chunk when party is this close to edge
        [SerializeField] private int maxGeneratedDepth = 50; // Maximum depth to generate (reduced to fit within grid)
        [SerializeField] private bool enableIncrementalGeneration = true;
        
        [Header("Events")]
        public UnityEvent<ExplorationNode> OnNodeDiscovered;
        public UnityEvent<ExplorationNode> OnNodeCompleted;
        public UnityEvent<Party, Vector2Int> OnPartyMoved; // New event for party movement
        public UnityEvent<Party> OnPartyDestroyed;
        public UnityEvent<Machine> OnMachineDestroyed;
        public UnityEvent<Machine> OnMachineRepaired;
        public UnityEvent<int> OnNewChunkGenerated; // New event for chunk generation
        public UnityEvent OnGraphInitialized; // New event for when the graph is fully generated
        
        // Core systems
        private Dictionary<Vector2Int, ExplorationNode> nodeGrid;
        private Dictionary<string, ExplorationNode> nodesById;
        private List<Party> parties;
        private int currentDepth;
        private bool isGraphGenerated;
        
        // Generation state
        private List<Vector2Int> pathNodes;
        private List<Vector2Int> branchNodes;
        
        // Incremental generation state
        private int lastGeneratedDepth;
        private Dictionary<int, bool> generatedChunks;
        private Queue<Vector2Int> pendingNodeConnections;
        
        void Awake()
        {
            InitializeSystem();
        }
        
        void Start()
        {
            GenerateInitialGraph();
            CreateStartingParties();
        }
        
        void Update()
        {
            if (enableIncrementalGeneration)
            {
                CheckForIncrementalGeneration();
            }
        }
        
        /// <summary>
        /// Initialize the graph system
        /// </summary>
        private void InitializeSystem()
        {
            nodeGrid = new Dictionary<Vector2Int, ExplorationNode>();
            nodesById = new Dictionary<string, ExplorationNode>();
            parties = new List<Party>();
            currentDepth = 0;
            isGraphGenerated = false;
            pathNodes = new List<Vector2Int>();
            branchNodes = new List<Vector2Int>();
            
            // Initialize incremental generation
            lastGeneratedDepth = 0;
            generatedChunks = new Dictionary<int, bool>();
            pendingNodeConnections = new Queue<Vector2Int>();
        }
        
        /// <summary>
        /// Generate the initial exploration graph
        /// </summary>
        private void GenerateInitialGraph()
        {
            if (isGraphGenerated) return;
            
            // Generate main path
            GenerateMainPath();
            
            // Generate branches
            GenerateBranches();
            
            // Generate special nodes
            GenerateSpecialNodes();
            
            // Connect nodes
            ConnectNodes();
            
            // Set starting node as accessible
            Vector2Int startPos = Vector2Int.zero;
            if (nodeGrid.ContainsKey(startPos))
            {
                nodeGrid[startPos].state = NodeState.Accessible;
                nodeGrid[startPos].type = NodeType.Start;
                
                Debug.Log($"GenerateInitialGraph: Set starting node at {startPos} to {NodeState.Accessible}");
                Debug.Log($"GenerateInitialGraph: Starting node connections: {string.Join(", ", nodeGrid[startPos].connectedNodeIds)}");
                
                // Reveal connected nodes to make them accessible for movement
                RevealConnectedNodes(nodeGrid[startPos]);
                
                // Debug: Check the final state of the starting node and its connections
                var startNode = nodeGrid[startPos];
                Debug.Log($"GenerateInitialGraph: Final starting node state: position={startNode.position}, type={startNode.type}, state={startNode.state}, connections={startNode.connectedNodeIds.Count}");
                
                foreach (var connectedId in startNode.connectedNodeIds)
                {
                    if (nodesById.ContainsKey(connectedId))
                    {
                        var connectedNode = nodesById[connectedId];
                        Debug.Log($"GenerateInitialGraph: Connected node {connectedId} at {connectedNode.position}: type={connectedNode.type}, state={connectedNode.state}");
                    }
                    else
                    {
                        Debug.LogWarning($"GenerateInitialGraph: Connected node ID {connectedId} not found in nodesById");
                    }
                }
            }
            else
            {
                Debug.LogError($"GenerateInitialGraph: Starting position {startPos} not found in nodeGrid!");
            }
            
            isGraphGenerated = true;
            Debug.Log($"Graph generated with {nodeGrid.Count} nodes");
            OnGraphInitialized?.Invoke();
            
            // Debug: Print some node states
            Debug.Log("GenerateInitialGraph: Sample node states:");
            int count = 0;
            foreach (var kvp in nodeGrid)
            {
                if (count < 10) // Only show first 10 nodes
                {
                    Debug.Log($"  Node at {kvp.Key}: type={kvp.Value.type}, state={kvp.Value.state}, connections={kvp.Value.connectedNodeIds.Count}");
                    count++;
                }
            }
        }
        
        /// <summary>
        /// Generate the main exploration path
        /// </summary>
        private void GenerateMainPath()
        {
            Vector2Int currentPos = Vector2Int.zero;
            pathNodes.Add(currentPos);
            
            // Create start node
            var startNode = new ExplorationNode(currentPos, NodeType.Start);
            AddNodeToGrid(startNode);
            Debug.Log($"GenerateMainPath: Created start node at {currentPos}");
            
            // Generate path to max depth
            for (int depth = 0; depth < maxDepth; depth++)
            {
                int pathsThisRow = Random.Range(minPathsPerRow, maxPathsPerRow + 1);
                Debug.Log($"GenerateMainPath: Depth {depth}, generating {pathsThisRow} paths");
                
                for (int path = 0; path < pathsThisRow; path++)
                {
                    Vector2Int nextPos = new Vector2Int(depth + 1, path);
                    pathNodes.Add(nextPos);
                    
                    NodeType nodeType = DetermineNodeType(depth, path, pathsThisRow);
                    var node = new ExplorationNode(nextPos, nodeType);
                    node.difficulty = CalculateDifficulty(depth);
                    node.rewardTier = CalculateRewardTier(depth);
                    
                    AddNodeToGrid(node);
                    Debug.Log($"GenerateMainPath: Created {nodeType} node at {nextPos}");
                }
            }
        }
        
        /// <summary>
        /// Generate branch nodes off the main path
        /// </summary>
        private void GenerateBranches()
        {
            foreach (var pathNode in pathNodes)
            {
                if (Random.value < 0.3f) // 30% chance to branch
                {
                    // Generate branch positions within valid grid bounds
                    int yOffset = Random.Range(-2, 3);
                    Vector2Int branchPos = pathNode + new Vector2Int(0, yOffset);
                    
                    // Ensure the branch position is valid before creating the node
                    if (!nodeGrid.ContainsKey(branchPos) && IsValidPosition(branchPos))
                    {
                        NodeType branchType = DetermineBranchNodeType(pathNode);
                        var branchNode = new ExplorationNode(branchPos, branchType);
                        branchNode.difficulty = CalculateDifficulty(pathNode.x);
                        branchNode.rewardTier = CalculateRewardTier(pathNode.x);
                        
                        AddNodeToGrid(branchNode);
                        branchNodes.Add(branchPos);
                        Debug.Log($"GenerateBranches: Created branch node at {branchPos} from path node at {pathNode}");
                    }
                    else
                    {
                        Debug.Log($"GenerateBranches: Skipped branch at {branchPos} - already exists or invalid position");
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate special nodes (bosses, shops, etc.)
        /// </summary>
        private void GenerateSpecialNodes()
        {
            // Generate boss nodes at certain depths
            for (int depth = 10; depth < maxDepth; depth += 10)
            {
                if (Random.value < bossNodeFrequency)
                {
                    // Ensure boss positions are within valid grid bounds
                    int yPos = Random.Range(0, Mathf.Min(3, gridHeight));
                    Vector2Int bossPos = new Vector2Int(depth, yPos);
                    if (nodeGrid.ContainsKey(bossPos))
                    {
                        nodeGrid[bossPos].type = NodeType.Boss;
                        nodeGrid[bossPos].difficulty = depth / 5;
                        Debug.Log($"GenerateSpecialNodes: Converted node at {bossPos} to Boss type");
                    }
                    else
                    {
                        Debug.Log($"GenerateSpecialNodes: Boss position {bossPos} not found in grid");
                    }
                }
            }
            
            // Generate shop nodes
            for (int depth = 5; depth < maxDepth; depth += 8)
            {
                if (Random.value < shopNodeFrequency)
                {
                    // Ensure shop positions are within valid grid bounds
                    int yPos = Random.Range(0, Mathf.Min(4, gridHeight));
                    Vector2Int shopPos = new Vector2Int(depth, yPos);
                    if (IsValidPosition(shopPos) && !nodeGrid.ContainsKey(shopPos))
                    {
                        var shopNode = new ExplorationNode(shopPos, NodeType.Shop);
                        AddNodeToGrid(shopNode);
                        Debug.Log($"GenerateSpecialNodes: Created shop node at {shopPos}");
                    }
                    else
                    {
                        Debug.Log($"GenerateSpecialNodes: Skipped shop at {shopPos} - already exists or invalid position");
                    }
                }
            }
        }
        
        /// <summary>
        /// Connect nodes to create the graph structure
        /// </summary>
        private void ConnectNodes()
        {
            Debug.Log("ConnectNodes: Starting to connect nodes...");
            
            foreach (var node in nodeGrid.Values)
            {
                // Find all nodes that could connect to this one
                var potentialConnections = GetPotentialConnections(node.position);
                
                Debug.Log($"ConnectNodes: Node at {node.position} has {potentialConnections.Count} potential connections");
                
                foreach (var connection in potentialConnections)
                {
                    if (nodeGrid.ContainsKey(connection))
                    {
                        // Bidirectional connection
                        if (!node.connectedNodeIds.Contains(nodeGrid[connection].nodeId))
                        {
                            node.connectedNodeIds.Add(nodeGrid[connection].nodeId);
                            Debug.Log($"ConnectNodes: Added connection from {node.position} to {connection}");
                        }
                        else
                        {
                            Debug.Log($"ConnectNodes: Connection from {node.position} to {connection} already exists");
                        }
                        if (!nodeGrid[connection].connectedNodeIds.Contains(node.nodeId))
                        {
                            nodeGrid[connection].connectedNodeIds.Add(node.nodeId);
                            Debug.Log($"ConnectNodes: Added connection from {connection} to {node.position}");
                        }
                        else
                        {
                            Debug.Log($"ConnectNodes: Connection from {connection} to {node.position} already exists");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"ConnectNodes: Potential connection position {connection} not found in nodeGrid");
                    }
                }
            }
            
            Debug.Log("ConnectNodes: Finished connecting nodes");
        }
        
        /// <summary>
        /// Get potential connection positions for a node
        /// </summary>
        private List<Vector2Int> GetPotentialConnections(Vector2Int pos)
        {
            var connections = new List<Vector2Int>();
            
            // Check adjacent positions
            Vector2Int[] directions = {
                new Vector2Int(1, 0),   // Right
                new Vector2Int(0, 1),   // Up
                new Vector2Int(0, -1),  // Down
                new Vector2Int(-1, 0)   // Left
            };
            
            Debug.Log($"GetPotentialConnections: Checking connections for position {pos}");
            
            foreach (var dir in directions)
            {
                Vector2Int checkPos = pos + dir;
                if (IsValidPosition(checkPos))
                {
                    connections.Add(checkPos);
                    Debug.Log($"GetPotentialConnections: Valid connection from {pos} to {checkPos}");
                }
                else
                {
                    Debug.Log($"GetPotentialConnections: Invalid connection from {pos} to {checkPos}");
                }
            }
            
            Debug.Log($"GetPotentialConnections: Found {connections.Count} valid connections for position {pos}");
            return connections;
        }
        
        /// <summary>
        /// Determine node type based on depth and position
        /// </summary>
        private NodeType DetermineNodeType(int depth, int pathIndex, int totalPaths)
        {
            if (depth < 3) return NodeType.Resource; // Early game resources
            
            float random = Random.value;
            
            if (depth % 10 == 0) return NodeType.Rest; // Rest points
            
            if (random < 0.4f) return NodeType.Combat;
            if (random < 0.6f) return NodeType.Resource;
            if (random < 0.8f) return NodeType.Event;
            if (random < 0.9f) return NodeType.Danger;
            return NodeType.Repair;
        }
        
        /// <summary>
        /// Determine branch node type
        /// </summary>
        private NodeType DetermineBranchNodeType(Vector2Int pathNode)
        {
            float random = Random.value;
            
            if (random < 0.3f) return NodeType.Resource;
            if (random < 0.5f) return NodeType.Event;
            if (random < 0.7f) return NodeType.Repair;
            if (random < 0.9f) return NodeType.Danger;
            return NodeType.Combat;
        }
        
        /// <summary>
        /// Calculate node difficulty based on depth
        /// </summary>
        private int CalculateDifficulty(int depth)
        {
            return Mathf.Max(1, depth / 5 + Random.Range(-1, 2));
        }
        
        /// <summary>
        /// Calculate reward tier based on depth
        /// </summary>
        private int CalculateRewardTier(int depth)
        {
            return Mathf.Max(1, depth / 8 + 1);
        }
        
        /// <summary>
        /// Check if a position is valid for node placement
        /// </summary>
        private bool IsValidPosition(Vector2Int pos)
        {
            bool isValid = pos.x >= 0 && pos.x < gridWidth && 
                          pos.y >= 0 && pos.y < gridHeight;
            
            Debug.Log($"IsValidPosition: Position {pos} is valid: {isValid} (gridWidth: {gridWidth}, gridHeight: {gridHeight})");
            return isValid;
        }
        
        /// <summary>
        /// Add a node to the grid system
        /// </summary>
        private void AddNodeToGrid(ExplorationNode node)
        {
            nodeGrid[node.position] = node;
            nodesById[node.nodeId] = node;
            Debug.Log($"AddNodeToGrid: Added {node.type} node at {node.position} with ID {node.nodeId}");
        }
        
        /// <summary>
        /// Create starting parties with basic machines
        /// </summary>
        private void CreateStartingParties()
        {
            // Create main party
            var mainParty = new Party("party_main", "Main Team");
            mainParty.members.Add(new Machine("rex", "Rex", 100, 25, 15));
            mainParty.members.Add(new Machine("luna", "Luna", 80, 30, 10));
            mainParty.members.Add(new Machine("zero", "Zero", 90, 20, 20));
            mainParty.currentNodePosition = Vector2Int.zero;
            parties.Add(mainParty);
            Debug.Log($"CreateStartingParties: Created main party at {mainParty.currentNodePosition}");
            
            // Create scout party
            var scoutParty = new Party("party_scout", "Scout Team");
            scoutParty.members.Add(new Machine("nova", "Nova", 70, 35, 8));
            scoutParty.members.Add(new Machine("echo", "Echo", 60, 40, 5));
            scoutParty.currentNodePosition = Vector2Int.zero;
            parties.Add(scoutParty);
            Debug.Log($"CreateStartingParties: Created scout party at {scoutParty.currentNodePosition}");
            
            Debug.Log($"Created {parties.Count} parties with {parties.Sum(p => p.members.Count)} total machines");
        }
        
        /// <summary>
        /// Move a party to a new node
        /// </summary>
        public bool MoveParty(string partyId, Vector2Int targetPosition)
        {
            Debug.Log($"MoveParty: Attempting to move party {partyId} to {targetPosition}");
            
            var party = parties.Find(p => p.id == partyId);
            if (party == null) 
            {
                Debug.LogError($"MoveParty: Party {partyId} not found");
                return false;
            }
            
            Debug.Log($"MoveParty: Party {partyId} found at {party.currentNodePosition}");
            
            if (!CanPartyMoveTo(party, targetPosition)) 
            {
                Debug.LogWarning($"MoveParty: CanPartyMoveTo returned false for party {partyId} to {targetPosition}");
                return false;
            }
            
            // Update party position
            party.currentNodePosition = targetPosition;
            Debug.Log($"MoveParty: Party {partyId} moved to {targetPosition}");
            
            // Trigger party movement event
            OnPartyMoved?.Invoke(party, targetPosition);
            
            // Discover the node
            if (nodeGrid.ContainsKey(targetPosition))
            {
                var node = nodeGrid[targetPosition];
                if (node.state == NodeState.Hidden)
                {
                    DiscoverNode(node);
                }
                
                // Make connected nodes visible
                RevealConnectedNodes(node);
            }
            
            return true;
        }
        
        /// <summary>
        /// Check if a party can move to a position
        /// </summary>
        private bool CanPartyMoveTo(Party party, Vector2Int targetPosition)
        {
            if (!nodeGrid.ContainsKey(targetPosition)) 
            {
                Debug.LogWarning($"CanPartyMoveTo: Target position {targetPosition} not found in grid");
                return false;
            }
            
            var currentNode = nodeGrid[party.currentNodePosition];
            var targetNode = nodeGrid[targetPosition];
            
            // Check if nodes are connected
            if (!currentNode.connectedNodeIds.Contains(targetNode.nodeId)) 
            {
                Debug.LogWarning($"CanPartyMoveTo: Nodes not connected. Current: {currentNode.position}, Target: {targetPosition}");
                Debug.LogWarning($"Current node connections: {string.Join(", ", currentNode.connectedNodeIds)}");
                return false;
            }
            
            // Check if target node is visible or accessible (not hidden)
            if (targetNode.state == NodeState.Hidden) 
            {
                Debug.LogWarning($"CanPartyMoveTo: Target node at {targetPosition} is hidden");
                return false;
            }
            
            // Check party energy
            if (party.energy < 10) 
            {
                Debug.LogWarning($"CanPartyMoveTo: Party {party.id} has insufficient energy: {party.energy}");
                return false;
            }
            
            Debug.Log($"CanPartyMoveTo: Party {party.id} can move from {currentNode.position} to {targetPosition}");
            return true;
        }
        
        /// <summary>
        /// Discover a hidden node
        /// </summary>
        private void DiscoverNode(ExplorationNode node)
        {
            node.state = NodeState.Visible;
            OnNodeDiscovered?.Invoke(node);
            Debug.Log($"Discovered {node.type} node at {node.position}");
        }
        
        /// <summary>
        /// Reveal nodes connected to a discovered node
        /// </summary>
        private void RevealConnectedNodes(ExplorationNode node)
        {
            Debug.Log($"RevealConnectedNodes: Revealing nodes connected to {node.position} (type: {node.type}, state: {node.state})");
            Debug.Log($"RevealConnectedNodes: Connected node IDs: {string.Join(", ", node.connectedNodeIds)}");
            
            foreach (var connectedId in node.connectedNodeIds)
            {
                if (nodesById.ContainsKey(connectedId))
                {
                    var connectedNode = nodesById[connectedId];
                    if (connectedNode.state == NodeState.Hidden)
                    {
                        connectedNode.state = NodeState.Visible;
                        Debug.Log($"RevealConnectedNodes: Changed node at {connectedNode.position} from Hidden to Visible");
                        
                        // Trigger discovery event for newly revealed nodes
                        OnNodeDiscovered?.Invoke(connectedNode);
                    }
                    else
                    {
                        Debug.Log($"RevealConnectedNodes: Node at {connectedNode.position} already in state: {connectedNode.state}");
                    }
                }
                else
                {
                    Debug.LogWarning($"RevealConnectedNodes: Connected node ID {connectedId} not found in nodesById");
                }
            }
        }
        
        /// <summary>
        /// Complete a node event
        /// </summary>
        public void CompleteNode(Vector2Int nodePosition)
        {
            if (!nodeGrid.ContainsKey(nodePosition)) return;
            
            var node = nodeGrid[nodePosition];
            node.state = NodeState.Completed;
            node.isVisited = true;
            
            OnNodeCompleted?.Invoke(node);
            
            // Make connected nodes accessible
            foreach (var connectedId in node.connectedNodeIds)
            {
                if (nodesById.ContainsKey(connectedId))
                {
                    var connectedNode = nodesById[connectedId];
                    if (connectedNode.state == NodeState.Visible)
                    {
                        connectedNode.state = NodeState.Accessible;
                    }
                }
            }
        }
        
        /// <summary>
        /// Get all accessible nodes for a party
        /// </summary>
        public List<ExplorationNode> GetAccessibleNodes(string partyId)
        {
            var party = parties.Find(p => p.id == partyId);
            if (party == null) return new List<ExplorationNode>();
            
            var accessibleNodes = new List<ExplorationNode>();
            var currentNode = nodeGrid[party.currentNodePosition];
            
            foreach (var connectedId in currentNode.connectedNodeIds)
            {
                if (nodesById.ContainsKey(connectedId))
                {
                    var connectedNode = nodesById[connectedId];
                    if (connectedNode.state == NodeState.Accessible)
                    {
                        accessibleNodes.Add(connectedNode);
                    }
                }
            }
            
            return accessibleNodes;
        }
        
        /// <summary>
        /// Get all visible nodes (for fog of war display)
        /// </summary>
        public List<ExplorationNode> GetVisibleNodes()
        {
            return nodeGrid.Values.Where(n => n.state != NodeState.Hidden).ToList();
        }
        
        /// <summary>
        /// Get all parties
        /// </summary>
        public List<Party> GetAllParties()
        {
            return parties;
        }
        
        /// <summary>
        /// Get a specific party by ID
        /// </summary>
        public Party GetParty(string partyId)
        {
            return parties.Find(p => p.id == partyId);
        }
        
        /// <summary>
        /// Get a node at a specific position
        /// </summary>
        public ExplorationNode GetNodeAt(Vector2Int position)
        {
            return nodeGrid.ContainsKey(position) ? nodeGrid[position] : null;
        }
        
        /// <summary>
        /// Check if the graph generation is complete
        /// </summary>
        public bool IsGraphGenerated()
        {
            return isGraphGenerated;
        }
        
        /// <summary>
        /// Get current exploration depth
        /// </summary>
        public int GetCurrentDepth()
        {
            return currentDepth;
        }
        
        /// <summary>
        /// Get total number of nodes
        /// </summary>
        public int GetTotalNodeCount()
        {
            return nodeGrid.Count;
        }
        
        /// <summary>
        /// Get completed node count
        /// </summary>
        public int GetCompletedNodeCount()
        {
            return nodeGrid.Values.Count(n => n.isVisited);
        }
        
        // ===== INCREMENTAL GENERATION METHODS =====
        
        /// <summary>
        /// Check if incremental generation is needed
        /// </summary>
        private void CheckForIncrementalGeneration()
        {
            if (!isGraphGenerated) return;
            
            // Check if any party is close to the generation threshold
            foreach (var party in parties)
            {
                if (party.currentNodePosition.x >= lastGeneratedDepth - generationThreshold)
                {
                    GenerateNextChunk();
                    break;
                }
            }
        }
        
        /// <summary>
        /// Generate the next chunk of the graph
        /// </summary>
        private void GenerateNextChunk()
        {
            if (lastGeneratedDepth >= maxGeneratedDepth) return;
            
            int nextChunkStart = lastGeneratedDepth;
            GenerateChunk(nextChunkStart, chunkSize);
            lastGeneratedDepth = nextChunkStart + chunkSize;
            
            Debug.Log($"Generated next chunk. Total depth now: {lastGeneratedDepth}");
        }
        
        /// <summary>
        /// Generate a chunk of the graph at the specified depth range
        /// </summary>
        private void GenerateChunk(int startDepth, int chunkDepth)
        {
            int chunkId = startDepth / chunkSize;
            if (generatedChunks.ContainsKey(chunkId)) return;
            
            Debug.Log($"Generating chunk {chunkId} from depth {startDepth} to {startDepth + chunkDepth}");
            
            // Generate main path for this chunk
            GenerateMainPathChunk(startDepth, chunkDepth);
            
            // Generate branches for this chunk
            GenerateBranchesChunk(startDepth, chunkDepth);
            
            // Generate special nodes for this chunk
            GenerateSpecialNodesChunk(startDepth, chunkDepth);
            
            // Connect nodes within this chunk
            ConnectNodesInChunk(startDepth, chunkDepth);
            
            // Connect to previous chunk if it exists
            if (startDepth > 0)
            {
                ConnectChunkToPrevious(startDepth);
            }
            
            generatedChunks[chunkId] = true;
            OnNewChunkGenerated?.Invoke(chunkId);
            
            Debug.Log($"Chunk {chunkId} generated with {GetNodeCountInDepthRange(startDepth, startDepth + chunkDepth)} nodes");
        }
        
        /// <summary>
        /// Generate main path for a specific chunk
        /// </summary>
        private void GenerateMainPathChunk(int startDepth, int chunkDepth)
        {
            for (int depth = startDepth; depth < startDepth + chunkDepth && depth < maxGeneratedDepth; depth++)
            {
                int pathsThisRow = Random.Range(minPathsPerRow, maxPathsPerRow + 1);
                
                for (int path = 0; path < pathsThisRow; path++)
                {
                    Vector2Int nextPos = new Vector2Int(depth, path);
                    if (!nodeGrid.ContainsKey(nextPos))
                    {
                        pathNodes.Add(nextPos);
                        
                        NodeType nodeType = DetermineNodeType(depth, path, pathsThisRow);
                        var node = new ExplorationNode(nextPos, nodeType);
                        node.difficulty = CalculateDifficulty(depth);
                        node.rewardTier = CalculateRewardTier(depth);
                        
                        AddNodeToGrid(node);
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate branches for a specific chunk
        /// </summary>
        private void GenerateBranchesChunk(int startDepth, int chunkDepth)
        {
            for (int depth = startDepth; depth < startDepth + chunkDepth && depth < maxGeneratedDepth; depth++)
            {
                var nodesAtDepth = GetNodesAtDepth(depth);
                foreach (var pathNode in nodesAtDepth)
                {
                    if (Random.value < 0.3f) // 30% chance to branch
                    {
                        Vector2Int branchPos = pathNode.position + new Vector2Int(0, Random.Range(-2, 3));
                        
                        if (!nodeGrid.ContainsKey(branchPos) && IsValidPosition(branchPos))
                        {
                            NodeType branchType = DetermineBranchNodeType(pathNode.position);
                            var branchNode = new ExplorationNode(branchPos, branchType);
                            branchNode.difficulty = CalculateDifficulty(depth);
                            branchNode.rewardTier = CalculateRewardTier(depth);
                            
                            AddNodeToGrid(branchNode);
                            branchNodes.Add(branchPos);
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate special nodes for a specific chunk
        /// </summary>
        private void GenerateSpecialNodesChunk(int startDepth, int chunkDepth)
        {
            for (int depth = startDepth; depth < startDepth + chunkDepth && depth < maxGeneratedDepth; depth++)
            {
                // Generate boss nodes at certain depths
                if (depth >= 10 && depth % 10 == 0 && Random.value < bossNodeFrequency)
                {
                    Vector2Int bossPos = new Vector2Int(depth, Random.Range(0, 3));
                    if (nodeGrid.ContainsKey(bossPos))
                    {
                        nodeGrid[bossPos].type = NodeType.Boss;
                        nodeGrid[bossPos].difficulty = depth / 5;
                    }
                }
                
                // Generate shop nodes
                if (depth >= 5 && depth % 8 == 0 && Random.value < shopNodeFrequency)
                {
                    Vector2Int shopPos = new Vector2Int(depth, Random.Range(-1, 4));
                    if (IsValidPosition(shopPos) && !nodeGrid.ContainsKey(shopPos))
                    {
                        var shopNode = new ExplorationNode(shopPos, NodeType.Shop);
                        AddNodeToGrid(shopNode);
                    }
                }
            }
        }
        
        /// <summary>
        /// Connect nodes within a specific chunk
        /// </summary>
        private void ConnectNodesInChunk(int startDepth, int chunkDepth)
        {
            for (int depth = startDepth; depth < startDepth + chunkDepth && depth < maxGeneratedDepth; depth++)
            {
                var nodesAtDepth = GetNodesAtDepth(depth);
                foreach (var node in nodesAtDepth)
                {
                    var potentialConnections = GetPotentialConnections(node.position);
                    
                    foreach (var connection in potentialConnections)
                    {
                        if (nodeGrid.ContainsKey(connection))
                        {
                            // Bidirectional connection
                            if (!node.connectedNodeIds.Contains(nodeGrid[connection].nodeId))
                            {
                                node.connectedNodeIds.Add(nodeGrid[connection].nodeId);
                            }
                            if (!nodeGrid[connection].connectedNodeIds.Contains(node.nodeId))
                            {
                                nodeGrid[connection].connectedNodeIds.Add(node.nodeId);
                            }
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Connect a chunk to the previous chunk
        /// </summary>
        private void ConnectChunkToPrevious(int startDepth)
        {
            var currentNodes = GetNodesAtDepth(startDepth);
            var previousNodes = GetNodesAtDepth(startDepth - 1);
            
            foreach (var currentNode in currentNodes)
            {
                foreach (var previousNode in previousNodes)
                {
                    // Connect if they're close enough
                    if (Vector2Int.Distance(currentNode.position, previousNode.position) <= 2)
                    {
                        if (!currentNode.connectedNodeIds.Contains(previousNode.nodeId))
                        {
                            currentNode.connectedNodeIds.Add(previousNode.nodeId);
                        }
                        if (!previousNode.connectedNodeIds.Contains(currentNode.nodeId))
                        {
                            previousNode.connectedNodeIds.Add(currentNode.nodeId);
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Get all nodes at a specific depth
        /// </summary>
        private List<ExplorationNode> GetNodesAtDepth(int depth)
        {
            return nodeGrid.Values.Where(n => n.position.x == depth).ToList();
        }
        
        /// <summary>
        /// Get node count in a depth range
        /// </summary>
        private int GetNodeCountInDepthRange(int startDepth, int endDepth)
        {
            return nodeGrid.Values.Count(n => n.position.x >= startDepth && n.position.x < endDepth);
        }
        
        /// <summary>
        /// Manually trigger generation of a specific chunk (for testing/debugging)
        /// </summary>
        public void GenerateChunkManually(int chunkId)
        {
            int startDepth = chunkId * chunkSize;
            if (startDepth >= maxGeneratedDepth) return;
            
            GenerateChunk(startDepth, chunkSize);
        }
        
        /// <summary>
        /// Get the current generation progress
        /// </summary>
        public float GetGenerationProgress()
        {
            if (maxGeneratedDepth <= 0) return 0f;
            return (float)lastGeneratedDepth / maxGeneratedDepth;
        }
        
        /// <summary>
        /// Get the last generated depth
        /// </summary>
        public int GetLastGeneratedDepth()
        {
            return lastGeneratedDepth;
        }
        
        /// <summary>
        /// Check if a specific chunk has been generated
        /// </summary>
        public bool IsChunkGenerated(int chunkId)
        {
            return generatedChunks.ContainsKey(chunkId) && generatedChunks[chunkId];
        }

        public void GenerateNextChunk()
        {
            if (lastGeneratedDepth >= maxGeneratedDepth) return;

            int nextChunkStart = lastGeneratedDepth;
            GenerateChunk(nextChunkStart, chunkSize);
            lastGeneratedDepth = nextChunkStart + chunkSize;

            Debug.Log($"Generated next chunk. Total depth now: {lastGeneratedDepth}");
        }

        public int GetLastGeneratedDepth()
        {
            return lastGeneratedDepth;
        }

        public int GetChunkSize()
        {
            return chunkSize;
        }

        public int GetMaxGeneratedDepth()
        {
            return maxGeneratedDepth;
        }
    }
}

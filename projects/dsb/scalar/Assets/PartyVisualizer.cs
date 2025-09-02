using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Scalar
{
    /// <summary>
    /// Visual representation of parties on the exploration graph
    /// </summary>
    public class PartyVisualizer : MonoBehaviour
    {
        [Header("Party Visual Settings")]
        [SerializeField] private GameObject partyPrefab;
        [SerializeField] private float partyScale = 1.0f;
        [SerializeField] private float heightOffset = 0.5f; // Slightly above nodes
        [SerializeField] private float gridSpacing = 3.0f; // Should match GraphVisualizer
        
        private GraphSystem graphSystem;
        private GraphVisualizer graphVisualizer;
        private CameraZoomSystem cameraZoomSystem;
        private Dictionary<string, GameObject> partyObjects;
        private Transform partyContainer;
        
        void Start()
        {
            graphSystem = FindFirstObjectByType<GraphSystem>();
            graphVisualizer = FindFirstObjectByType<GraphVisualizer>();
            cameraZoomSystem = FindFirstObjectByType<CameraZoomSystem>();
            
            if (graphSystem == null)
            {
                Debug.LogError("PartyVisualizer: No GraphSystem found in scene!");
                return;
            }
            
            InitializePartyVisualizer();
            
            // Subscribe to party movement events
            graphSystem.OnPartyMoved.AddListener(OnPartyMoved);
            
            // Get grid spacing from GraphVisualizer to ensure consistent positioning
            if (graphVisualizer != null)
            {
                gridSpacing = graphVisualizer.GridSpacing;
                Debug.Log($"PartyVisualizer: Using grid spacing {gridSpacing} from GraphVisualizer");
            }
        }
        
        void Update()
        {
            // Update party positions
            UpdatePartyPositions();
        }
        
        /// <summary>
        /// Initialize the party visualizer
        /// </summary>
        private void InitializePartyVisualizer()
        {
            partyObjects = new Dictionary<string, GameObject>();
            partyContainer = new GameObject("PartyContainer").transform;
            partyContainer.SetParent(transform);
            
            // Create visuals for existing parties
            CreateAllPartyVisuals();
        }
        
        /// <summary>
        /// Create visuals for all existing parties
        /// </summary>
        private void CreateAllPartyVisuals()
        {
            if (graphSystem == null) return;
            
            var allParties = graphSystem.GetAllParties();
            foreach (var party in allParties)
            {
                CreatePartyVisual(party);
            }
        }
        
        /// <summary>
        /// Create visual representation of a party
        /// </summary>
        private void CreatePartyVisual(Party party)
        {
            if (partyObjects.ContainsKey(party.id)) return;
            
            Vector3 worldPos = GetPartyWorldPosition(party.currentNodePosition);
            Debug.Log($"PartyVisualizer: Creating party {party.id} at grid position {party.currentNodePosition} -> world position {worldPos}");
            
            GameObject partyObj;
            
            if (partyPrefab != null)
            {
                partyObj = Instantiate(partyPrefab, worldPos, Quaternion.identity);
            }
            else
            {
                // Create default party representation - a distinctive group of objects
                partyObj = new GameObject($"Party_{party.name}_{party.id}");
                partyObj.transform.position = worldPos;
                
                // Create main party body (larger sphere)
                var mainBody = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                mainBody.transform.SetParent(partyObj.transform);
                mainBody.transform.localPosition = Vector3.zero;
                mainBody.transform.localScale = Vector3.one * 0.8f;
                
                // Set color based on party status
                var renderer = mainBody.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = party.isActive ? Color.green : Color.gray;
                }
                
                // Create party members (smaller spheres around the main body)
                int memberCount = Mathf.Min(party.members.Count, 4); // Max 4 visible members
                for (int i = 0; i < memberCount; i++)
                {
                    var member = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    member.transform.SetParent(partyObj.transform);
                    
                    // Position members in a circle around the main body
                    float angle = (i * 360f / memberCount) * Mathf.Deg2Rad;
                    float radius = 0.6f;
                    member.transform.localPosition = new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
                    member.transform.localScale = Vector3.one * 0.4f;
                    
                    // Set member color (slightly different from main body)
                    var memberRenderer = member.GetComponent<Renderer>();
                    if (memberRenderer != null)
                    {
                        memberRenderer.material.color = party.isActive ? Color.cyan : Color.darkGray;
                    }
                }
                
                // Add a light to make the party more visible
                var light = partyObj.AddComponent<Light>();
                light.type = LightType.Point;
                light.range = 2f;
                light.intensity = 1f;
                light.color = party.isActive ? Color.green : Color.gray;
            }
            
            partyObj.transform.SetParent(partyContainer);
            partyObj.transform.localScale = Vector3.one * partyScale;
            partyObj.name = $"Party_{party.name}_{party.id}";
            
            // Add party info component
            var partyInfo = partyObj.AddComponent<PartyInfo>();
            partyInfo.Initialize(party);
            
            partyObjects[party.id] = partyObj;
            Debug.Log($"PartyVisualizer: Created party object {partyObj.name} at {worldPos}");
        }
        
        /// <summary>
        /// Update positions of all party objects
        /// </summary>
        private void UpdatePartyPositions()
        {
            if (graphSystem == null) return;
            
            var allParties = graphSystem.GetAllParties();
            foreach (var party in allParties)
            {
                if (partyObjects.ContainsKey(party.id))
                {
                    var partyObj = partyObjects[party.id];
                    Vector3 targetPos = GetPartyWorldPosition(party.currentNodePosition);
                    
                    // Smoothly move party to new position
                    if (Vector3.Distance(partyObj.transform.position, targetPos) > 0.1f)
                    {
                        partyObj.transform.position = Vector3.Lerp(partyObj.transform.position, targetPos, Time.deltaTime * 5.0f);
                    }
                }
                else
                {
                    // Create visual for new party
                    CreatePartyVisual(party);
                }
            }
        }
        
        /// <summary>
        /// Get world position for a party at a grid position
        /// </summary>
        private Vector3 GetPartyWorldPosition(Vector2Int gridPos)
        {
            return new Vector3(gridPos.x * gridSpacing, heightOffset, gridPos.y * gridSpacing);
        }
        
        /// <summary>
        /// Center camera on a specific party using CameraZoomSystem
        /// </summary>
        public void CenterCameraOnParty(string partyId)
        {
            if (graphSystem == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);
            
            if (cameraZoomSystem != null)
            {
                cameraZoomSystem.CenterOnPosition(partyWorldPos);
                Debug.Log($"PartyVisualizer: Centering camera on party {partyId} at {party.currentNodePosition} using CameraZoomSystem");
            }
            else
            {
                Debug.LogWarning("PartyVisualizer: No CameraZoomSystem found, cannot center camera.");
            }
        }
        
        /// <summary>
        /// Center camera on a specific party and make it look at the party using CameraZoomSystem
        /// </summary>
        public void CenterCameraOnPartyAndLookAt(string partyId)
        {
            if (graphSystem == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);

            if (cameraZoomSystem != null)
            {
                cameraZoomSystem.CenterOnPosition(partyWorldPos);
                Debug.Log($"PartyVisualizer: Centering camera on party {partyId} at {party.currentNodePosition} using CameraZoomSystem");
            }
            else
            {
                Debug.LogWarning("PartyVisualizer: No CameraZoomSystem found, cannot center camera.");
            }
        }
        
        
        
        /// <summary>
        /// Get party object by ID
        /// </summary>
        public GameObject GetPartyObject(string partyId)
        {
            return partyObjects.ContainsKey(partyId) ? partyObjects[partyId] : null;
        }
        
        /// <summary>
        /// Clear all party visuals
        /// </summary>
        public void ClearPartyVisuals()
        {
            if (partyContainer != null)
            {
                DestroyImmediate(partyContainer.gameObject);
            }
            
            partyObjects.Clear();
            InitializePartyVisualizer();
        }
        
        /// <summary>
        /// Force update all party positions
        /// </summary>
        public void ForceUpdatePartyPositions()
        {
            Debug.Log("PartyVisualizer: Force updating party positions...");
            UpdatePartyPositions();
        }
        
        /// <summary>
        /// Handle party movement event
        /// </summary>
        private void OnPartyMoved(Party party, Vector2Int newPosition)
        {
            Debug.Log($"PartyVisualizer: Party {party.id} moved to {newPosition}");
            
            // Update party position immediately
            if (partyObjects.ContainsKey(party.id))
            {
                var partyObj = partyObjects[party.id];
                Vector3 targetPos = GetPartyWorldPosition(newPosition);
                partyObj.transform.position = targetPos;
            }
            
            // Camera movement is now handled by CameraZoomSystem
        }
        
        /// <summary>
        /// Update party visual state (active/inactive, destroyed, etc.)
        /// </summary>
        public void UpdatePartyState(string partyId)
        {
            if (graphSystem == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            if (partyObjects.ContainsKey(partyId))
            {
                var partyObj = partyObjects[partyId];
                
                // Update colors based on party state
                var renderers = partyObj.GetComponentsInChildren<Renderer>();
                foreach (var renderer in renderers)
                {
                    if (renderer.name.Contains("Sphere"))
                    {
                        if (renderer.transform.parent == partyObj.transform)
                        {
                            // Main body
                            renderer.material.color = party.isActive ? Color.green : Color.gray;
                        }
                        else
                        {
                            // Party members
                            renderer.material.color = party.isActive ? Color.cyan : Color.darkGray;
                        }
                    }
                }
                
                // Update light
                var light = partyObj.GetComponent<Light>();
                if (light != null)
                {
                    light.color = party.isActive ? Color.green : Color.gray;
                    light.intensity = party.isActive ? 1f : 0.3f;
                }
                
                Debug.Log($"PartyVisualizer: Updated party {partyId} state - Active: {party.isActive}");
            }
        }
        
        
            
    }
    
    /// <summary>
    /// Component attached to party objects to store party information
    /// </summary>
    public class PartyInfo : MonoBehaviour
    {
        public Party party;
        
        public void Initialize(Party partyData)
        {
            party = partyData;
        }
        
        void OnMouseDown()
        {
            // Handle party selection
            Debug.Log($"Selected party: {party.name} at {party.currentNodePosition}");
            
            // Center camera on this party
            var partyVisualizer = FindFirstObjectByType<PartyVisualizer>();
            if (partyVisualizer != null)
            {
                partyVisualizer.CenterCameraOnParty(party.id);
            }
        }
        
        void OnMouseEnter()
        {
            // Show party tooltip
            transform.localScale *= 1.2f;
        }
        
        void OnMouseExit()
        {
            // Hide party tooltip
            transform.localScale /= 1.2f;
        }
    }
}

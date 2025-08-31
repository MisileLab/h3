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
        
        [Header("Camera Settings")]
        [SerializeField] private Camera targetCamera;
        [SerializeField] private float cameraMoveSpeed = 5.0f;
        [SerializeField] private float cameraHeight = 15.0f; // Increased for better overhead view
        [SerializeField] private float cameraDistance = 0.0f; // Distance from party (0 = directly above)
        [SerializeField] private bool autoCenterCamera = true;
        [SerializeField] private float autoCenterDelay = 0.5f; // Delay before auto-centering
        [SerializeField] private Vector3 cameraOffset = new Vector3(0, 0, 0); // Additional offset for fine-tuning
        
        private GraphSystem graphSystem;
        private GraphVisualizer graphVisualizer;
        private Dictionary<string, GameObject> partyObjects;
        private Transform partyContainer;
        private Vector3 targetCameraPosition;
        private bool isMovingCamera = false;
        private float lastPartyMoveTime;
        
        void Start()
        {
            graphSystem = FindFirstObjectByType<GraphSystem>();
            graphVisualizer = FindFirstObjectByType<GraphVisualizer>();
            
            if (graphSystem == null)
            {
                Debug.LogError("PartyVisualizer: No GraphSystem found in scene!");
                return;
            }
            
            if (targetCamera == null)
            {
                targetCamera = Camera.main;
                if (targetCamera == null)
                {
                    Debug.LogError("PartyVisualizer: No camera found!");
                    return;
                }
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
            
            // Handle camera movement
            if (isMovingCamera)
            {
                MoveCameraToTarget();
            }
            
            // Auto-center camera after party movement
            if (autoCenterCamera && Time.time - lastPartyMoveTime > autoCenterDelay)
            {
                CenterCameraOnActiveParty();
            }
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
                        lastPartyMoveTime = Time.time;
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
        /// Center camera on a specific party
        /// </summary>
        public void CenterCameraOnParty(string partyId)
        {
            if (graphSystem == null || targetCamera == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);
            
            // Calculate camera position: directly above the party with optional offset
            Vector3 cameraPos = new Vector3(
                partyWorldPos.x + cameraOffset.x,
                partyWorldPos.y + cameraHeight + cameraOffset.y,
                partyWorldPos.z + cameraOffset.z
            );
            
            targetCameraPosition = cameraPos;
            isMovingCamera = true;
            
            Debug.Log($"PartyVisualizer: Centering camera on party {partyId} at {party.currentNodePosition}");
            Debug.Log($"PartyVisualizer: Party world position: {partyWorldPos}, Camera target: {cameraPos}");
        }
        
        /// <summary>
        /// Center camera on a specific party and make it look at the party
        /// </summary>
        public void CenterCameraOnPartyAndLookAt(string partyId)
        {
            if (graphSystem == null || targetCamera == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);
            
            // Calculate camera position: directly above the party with optional offset
            Vector3 cameraPos = new Vector3(
                partyWorldPos.x + cameraOffset.x,
                partyWorldPos.y + cameraHeight + cameraOffset.y,
                partyWorldPos.z + cameraOffset.z
            );
            
            targetCameraPosition = cameraPos;
            isMovingCamera = true;
            
            // Make camera look at the party
            StartCoroutine(LookAtPartyWhenCameraStops(partyWorldPos));
            
            Debug.Log($"PartyVisualizer: Centering camera on party {partyId} at {party.currentNodePosition} and will look at party");
            Debug.Log($"PartyVisualizer: Party world position: {partyWorldPos}, Camera target: {cameraPos}");
        }
        
        /// <summary>
        /// Center camera on the active party
        /// </summary>
        public void CenterCameraOnActiveParty()
        {
            if (graphSystem == null) return;
            
            var activeParty = graphSystem.GetAllParties().Find(p => p.isActive);
            if (activeParty != null)
            {
                CenterCameraOnParty(activeParty.id);
            }
        }
        
        /// <summary>
        /// Move camera to target position smoothly
        /// </summary>
        private void MoveCameraToTarget()
        {
            if (targetCamera == null) return;
            
            float distance = Vector3.Distance(targetCamera.transform.position, targetCameraPosition);
            if (distance > 0.1f)
            {
                targetCamera.transform.position = Vector3.Lerp(targetCamera.transform.position, targetCameraPosition, Time.deltaTime * cameraMoveSpeed);
            }
            else
            {
                isMovingCamera = false;
                Debug.Log("PartyVisualizer: Camera reached target position");
            }
        }
        
        /// <summary>
        /// Set camera target position directly
        /// </summary>
        public void SetCameraTarget(Vector3 targetPos)
        {
            targetCameraPosition = targetPos;
            isMovingCamera = true;
        }
        
        /// <summary>
        /// Toggle auto-camera centering
        /// </summary>
        public void ToggleAutoCenterCamera()
        {
            autoCenterCamera = !autoCenterCamera;
            Debug.Log($"PartyVisualizer: Auto-camera centering {(autoCenterCamera ? "enabled" : "disabled")}");
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
                lastPartyMoveTime = Time.time;
            }
            
            // Center camera on the moved party if auto-centering is enabled
            if (autoCenterCamera)
            {
                CenterCameraOnParty(party.id);
            }
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
        
        /// <summary>
        /// Center camera on the current position of a specific party
        /// </summary>
        public void CenterCameraOnPartyCurrentPosition(string partyId)
        {
            if (graphSystem == null || targetCamera == null) return;
            
            var party = graphSystem.GetParty(partyId);
            if (party == null) return;
            
            Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);
            
            // Calculate camera position: directly above the party with optional offset
            Vector3 cameraPos = new Vector3(
                partyWorldPos.x + cameraOffset.x,
                partyWorldPos.y + cameraHeight + cameraOffset.y,
                partyWorldPos.z + cameraOffset.z
            );
            
            targetCameraPosition = cameraPos;
            isMovingCamera = true;
            
            Debug.Log($"PartyVisualizer: Centering camera on party {partyId} current position {party.currentNodePosition}");
            Debug.Log($"PartyVisualizer: Party world position: {partyWorldPos}, Camera target: {cameraPos}");
        }
        
        /// <summary>
        /// Get the current camera target position
        /// </summary>
        public Vector3 GetCurrentCameraTarget()
        {
            return targetCameraPosition;
        }
        
        /// <summary>
        /// Check if camera is currently moving
        /// </summary>
        public bool IsCameraMoving()
        {
            return isMovingCamera;
        }
        
        /// <summary>
        /// Coroutine to make camera look at party when it stops moving
        /// </summary>
        private IEnumerator LookAtPartyWhenCameraStops(Vector3 partyPosition)
        {
            // Wait for camera to stop moving
            while (isMovingCamera)
            {
                yield return null;
            }
            
            // Make camera look at the party
            if (targetCamera != null)
            {
                targetCamera.transform.LookAt(partyPosition);
                Debug.Log("PartyVisualizer: Camera now looking at party");
            }
        }
        
        /// <summary>
        /// Debug method to show current camera and party positions
        /// </summary>
        public void DebugCameraPositions()
        {
            if (graphSystem == null || targetCamera == null) return;
            
            var allParties = graphSystem.GetAllParties();
            Debug.Log("=== Camera Position Debug ===");
            Debug.Log($"Current camera position: {targetCamera.transform.position}");
            Debug.Log($"Current camera target: {targetCameraPosition}");
            Debug.Log($"Camera is moving: {isMovingCamera}");
            
            foreach (var party in allParties)
            {
                Vector3 partyWorldPos = GetPartyWorldPosition(party.currentNodePosition);
                Vector3 cameraPos = new Vector3(
                    partyWorldPos.x + cameraOffset.x,
                    partyWorldPos.y + cameraHeight + cameraOffset.y,
                    partyWorldPos.z + cameraOffset.z
                );
                
                Debug.Log($"Party {party.id} at grid {party.currentNodePosition} -> world {partyWorldPos} -> camera target {cameraPos}");
            }
            Debug.Log("=== End Debug ===");
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

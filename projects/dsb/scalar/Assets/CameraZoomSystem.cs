using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections;

namespace Scalar
{
    /// <summary>
    /// Handles camera zoom and movement controls
    /// </summary>
    public class CameraZoomSystem : MonoBehaviour
    {
        [Header("Camera Reference")]
        [SerializeField] private Camera targetCamera;
        
        [Header("Zoom Settings")]
        [SerializeField] private float zoomSpeed = 5f;
        [SerializeField] private float minZoom = 1f;
        [SerializeField] private float maxZoom = 20f;
        [SerializeField] private float zoomSmoothness = 5f;
        
        [Header("Movement Settings")]
        [SerializeField] private float moveSpeed = 10f;
        [SerializeField] private float panSpeed = 15f;
        [SerializeField] private float edgeScrollThreshold = 20f;
        [SerializeField] private bool enableEdgeScrolling = true;
        [SerializeField] private bool enableWASD = false;
        [SerializeField] private bool enableMousePan = true;
        
        [Header("Boundaries")]
        [SerializeField] private bool enableBoundaries = true;
        [SerializeField] private float minX = -50f;
        [SerializeField] private float maxX = 50f;
        [SerializeField] private float minZ = -50f;
        [SerializeField] private float maxZ = 50f;
        
        [Header("Target Following")]
        [SerializeField] private Transform followTarget;
        [SerializeField] private Vector3 followOffset = new Vector3(0, 15, 0);
        [SerializeField] private float followSpeed = 5f;
        
        private Vector3 targetPosition;
        private float targetZoom;
        private bool isFollowing = false;
        private Vector3 lastMousePosition;
        private bool isPanning = false;
        
        // Input System references
        private PlayerInput playerInput;
        private InputAction moveAction;
        private InputAction zoomAction;
        private InputAction panAction;
        
        void Start()
        {
            if (targetCamera == null)
            {
                targetCamera = Camera.main;
                if (targetCamera == null)
                {
                    Debug.LogError("CameraZoomSystem: No camera found!");
                    return;
                }
            }
            
            // Initialize target values
            targetPosition = targetCamera.transform.position;
            targetZoom = targetCamera.orthographic ? targetCamera.orthographicSize : targetCamera.fieldOfView;
            
            // Setup Input System
            SetupInputSystem();
            
            // Focus on main party after a short delay to ensure everything is initialized
            StartCoroutine(FocusOnMainPartyDelayed());
            
            Debug.Log("CameraZoomSystem: Initialized successfully");
        }
        
        /// <summary>
        /// Focus on the main party after initialization
        /// </summary>
        private IEnumerator FocusOnMainPartyDelayed()
        {
            // Wait a frame to ensure GraphSystem and PartyVisualizer are initialized
            yield return null;
            
            // Wait additional time to ensure GraphVisualizer has created its visualization
            Debug.Log("CameraZoomSystem: Waiting for GraphVisualizer to create visualization...");
            yield return new WaitForSeconds(1.0f);
            
            // Try to find and focus on the main party
            Debug.Log("CameraZoomSystem: Ready to focus on main party");
            FocusOnMainParty();
        }
        
        /// <summary>
        /// Focus on the main party
        /// </summary>
        private void FocusOnMainParty()
        {
            var graphSystem = FindFirstObjectByType<GraphSystem>();
            if (graphSystem == null)
            {
                Debug.LogWarning("CameraZoomSystem: No GraphSystem found, cannot focus on main party");
                return;
            }
            
            var mainParty = graphSystem.GetParty("party_main");
            if (mainParty == null)
            {
                Debug.LogWarning("CameraZoomSystem: Main party not found, cannot focus on main party");
                return;
            }
            
            // Convert grid position to world position
            Vector3 worldPos = ConvertGridToWorldPosition(mainParty.currentNodePosition);
            
            // Center camera on main party
            CenterOnPosition(worldPos);
            
            Debug.Log($"CameraZoomSystem: Focused on main party at grid position {mainParty.currentNodePosition} -> world position {worldPos}");
        }
        
        /// <summary>
        /// Convert grid position to world position
        /// </summary>
        private Vector3 ConvertGridToWorldPosition(Vector2Int gridPos)
        {
            // This should match the grid spacing used in GraphVisualizer and PartyVisualizer
            float gridSpacing = 3.0f; // Default grid spacing
            
            // Try to get grid spacing from GraphVisualizer if available
            var graphVisualizer = FindFirstObjectByType<GraphVisualizer>();
            if (graphVisualizer != null)
            {
                gridSpacing = graphVisualizer.GridSpacing;
            }
            
            return new Vector3(gridPos.x * gridSpacing, 0, gridPos.y * gridSpacing);
        }
        
        /// <summary>
        /// Public method to focus on main party
        /// </summary>
        public void FocusOnMainPartyPublic()
        {
            FocusOnMainParty();
        }
        
        /// <summary>
        /// Focus on a specific party by ID
        /// </summary>
        public void FocusOnParty(string partyId)
        {
            var graphSystem = FindFirstObjectByType<GraphSystem>();
            if (graphSystem == null)
            {
                Debug.LogWarning("CameraZoomSystem: No GraphSystem found, cannot focus on party");
                return;
            }
            
            var party = graphSystem.GetParty(partyId);
            if (party == null)
            {
                Debug.LogWarning($"CameraZoomSystem: Party {partyId} not found, cannot focus on party");
                return;
            }
            
            // Convert grid position to world position
            Vector3 worldPos = ConvertGridToWorldPosition(party.currentNodePosition);
            
            // Center camera on party
            CenterOnPosition(worldPos);
            
            Debug.Log($"CameraZoomSystem: Focused on party {partyId} at grid position {party.currentNodePosition} -> world position {worldPos}");
        }
        
        /// <summary>
        /// Setup the Input System actions
        /// </summary>
        private void SetupInputSystem()
        {
            // Try to get PlayerInput component
            playerInput = GetComponent<PlayerInput>();
            
            if (playerInput == null)
            {
                // Create actions manually if no PlayerInput component
                CreateInputActions();
            }
        }
        
        /// <summary>
        /// Create Input Actions manually
        /// </summary>
        private void CreateInputActions()
        {
            // Create action map
            var actionMap = new InputActionMap("Camera");
            
            // Create actions
            moveAction = actionMap.AddAction("Move", InputActionType.Value, "<Keyboard>/w,<Keyboard>/s,<Keyboard>/a,<Keyboard>/d,<Keyboard>/upArrow,<Keyboard>/downArrow,<Keyboard>/leftArrow,<Keyboard>/rightArrow");
            zoomAction = actionMap.AddAction("Zoom", InputActionType.Value, "<Mouse>/scroll");
            panAction = actionMap.AddAction("Pan", InputActionType.Button, "<Mouse>/middleButton");
            
            // Enable actions
            moveAction.Enable();
            zoomAction.Enable();
            panAction.Enable();
            
            Debug.Log("CameraZoomSystem: Created manual input actions");
        }
        
        void Update()
        {
            HandleInput();
            UpdateCamera();
        }
        
        /// <summary>
        /// Handle all input for camera control
        /// </summary>
        private void HandleInput()
        {
            // Zoom input
            HandleZoomInput();
            
            // Movement input
            HandleMovementInput();
            
            // Mouse pan input
            HandleMousePanInput();
        }
        
        /// <summary>
        /// Handle zoom input from mouse wheel
        /// </summary>
        private void HandleZoomInput()
        {
            if (zoomAction != null)
            {
                float scroll = zoomAction.ReadValue<Vector2>().y;
                if (Mathf.Abs(scroll) > 0.01f)
                {
                    targetZoom -= scroll * zoomSpeed;
                    targetZoom = Mathf.Clamp(targetZoom, minZoom, maxZoom);
                }
            }
        }
        
        /// <summary>
        /// Handle movement input from keyboard and edge scrolling
        /// </summary>
        private void HandleMovementInput()
        {
            Vector3 moveDirection = Vector3.zero;
            
            // WASD movement
            if (enableWASD && moveAction != null)
            {
                Vector2 moveInput = moveAction.ReadValue<Vector2>();
                if (moveInput.magnitude > 0.01f)
                {
                    moveDirection += new Vector3(moveInput.x, 0, moveInput.y);
                }
            }
            
            // Edge scrolling
            if (enableEdgeScrolling)
            {
                Vector3 mousePos = Mouse.current.position.ReadValue();
                
                if (mousePos.x < edgeScrollThreshold)
                    moveDirection += Vector3.left;
                else if (mousePos.x > Screen.width - edgeScrollThreshold)
                    moveDirection += Vector3.right;
                    
                if (mousePos.y < edgeScrollThreshold)
                    moveDirection += Vector3.back;
                else if (mousePos.y > Screen.height - edgeScrollThreshold)
                    moveDirection += Vector3.forward;
            }
            
            // Apply movement
            if (moveDirection.magnitude > 0.01f)
            {
                StopFollowing();
                Vector3 worldMove = targetCamera.transform.TransformDirection(moveDirection);
                worldMove.y = 0; // Keep camera at same height
                targetPosition += worldMove * moveSpeed * Time.deltaTime;
            }
        }
        
        /// <summary>
        /// Handle mouse pan input (middle mouse button)
        /// </summary>
        private void HandleMousePanInput()
        {
            if (!enableMousePan) return;
            
            // Start panning
            if (panAction != null && panAction.WasPressedThisFrame())
            {
                isPanning = true;
                lastMousePosition = Mouse.current.position.ReadValue();
                StopFollowing();
            }
            
            // Panning
            if (isPanning && panAction != null && panAction.IsPressed())
            {
                Vector3 currentMousePos = Mouse.current.position.ReadValue();
                Vector3 delta = currentMousePos - lastMousePosition;
                Vector3 worldDelta = targetCamera.transform.TransformDirection(new Vector3(-delta.x, 0, -delta.y));
                worldDelta.y = 0; // Keep camera at same height
                targetPosition += worldDelta * panSpeed * Time.deltaTime;
                lastMousePosition = currentMousePos;
            }
            
            // Stop panning
            if (panAction != null && panAction.WasReleasedThisFrame())
            {
                isPanning = false;
            }
        }
        
        /// <summary>
        /// Update camera position and zoom smoothly
        /// </summary>
        private void UpdateCamera()
        {
            // Handle target following
            if (isFollowing && followTarget != null)
            {
                Vector3 followPosition = followTarget.position + followOffset;
                targetPosition = Vector3.Lerp(targetPosition, followPosition, followSpeed * Time.deltaTime);
            }
            
            // Apply boundaries
            if (enableBoundaries)
            {
                targetPosition.x = Mathf.Clamp(targetPosition.x, minX, maxX);
                targetPosition.z = Mathf.Clamp(targetPosition.z, minZ, maxZ);
            }
            
            // Smoothly move camera to target position
            if (Vector3.Distance(targetCamera.transform.position, targetPosition) > 0.01f)
            {
                targetCamera.transform.position = Vector3.Lerp(
                    targetCamera.transform.position, 
                    targetPosition, 
                    Time.deltaTime * moveSpeed
                );
            }
            
            // Smoothly apply zoom
            if (targetCamera.orthographic)
            {
                if (Mathf.Abs(targetCamera.orthographicSize - targetZoom) > 0.01f)
                {
                    targetCamera.orthographicSize = Mathf.Lerp(
                        targetCamera.orthographicSize, 
                        targetZoom, 
                        Time.deltaTime * zoomSmoothness
                    );
                }
            }
            else
            {
                if (Mathf.Abs(targetCamera.fieldOfView - targetZoom) > 0.01f)
                {
                    targetCamera.fieldOfView = Mathf.Lerp(
                        targetCamera.fieldOfView, 
                        targetZoom, 
                        Time.deltaTime * zoomSmoothness
                    );
                }
            }
        }
        
        /// <summary>
        /// Set camera to follow a target
        /// </summary>
        public void FollowTarget(Transform target)
        {
            followTarget = target;
            isFollowing = true;
            Debug.Log($"CameraZoomSystem: Now following target {target.name}");
        }
        
        /// <summary>
        /// Stop following the current target
        /// </summary>
        public void StopFollowing()
        {
            isFollowing = false;
            Debug.Log("CameraZoomSystem: Stopped following target");
        }
        
        /// <summary>
        /// Set camera position instantly
        /// </summary>
        public void SetPosition(Vector3 position)
        {
            targetPosition = position;
            targetCamera.transform.position = position;
            StopFollowing();
            Debug.Log($"CameraZoomSystem: Camera position set to {position}");
        }
        
        /// <summary>
        /// Move camera to position smoothly
        /// </summary>
        public void MoveToPosition(Vector3 position)
        {
            targetPosition = position;
            StopFollowing();
            Debug.Log($"CameraZoomSystem: Camera moving to {position}");
        }
        
        /// <summary>
        /// Center camera on a specific world position
        /// </summary>
        public void CenterOnPosition(Vector3 worldPosition)
        {
            Vector3 centerPosition = new Vector3(worldPosition.x, targetCamera.transform.position.y, worldPosition.z);
            MoveToPosition(centerPosition);
            Debug.Log($"CameraZoomSystem: Centering camera on {worldPosition}");
        }
        
        /// <summary>
        /// Set zoom level instantly
        /// </summary>
        public void SetZoom(float zoom)
        {
            targetZoom = Mathf.Clamp(zoom, minZoom, maxZoom);
            if (targetCamera.orthographic)
                targetCamera.orthographicSize = targetZoom;
            else
                targetCamera.fieldOfView = targetZoom;
            Debug.Log($"CameraZoomSystem: Zoom set to {zoom}");
        }
        
        /// <summary>
        /// Reset camera to default position and zoom
        /// </summary>
        public void ResetCamera()
        {
            Vector3 defaultPosition = new Vector3(0, 15, -10);
            float defaultZoom = targetCamera.orthographic ? 5f : 60f;
            
            SetPosition(defaultPosition);
            SetZoom(defaultZoom);
            StopFollowing();
            Debug.Log("CameraZoomSystem: Camera reset to default");
        }
        
        /// <summary>
        /// Toggle edge scrolling
        /// </summary>
        public void ToggleEdgeScrolling()
        {
            enableEdgeScrolling = !enableEdgeScrolling;
            Debug.Log($"CameraZoomSystem: Edge scrolling {(enableEdgeScrolling ? "enabled" : "disabled")}");
        }
        
        /// <summary>
        /// Toggle WASD movement
        /// </summary>
        public void ToggleWASD()
        {
            enableWASD = !enableWASD;
            Debug.Log($"CameraZoomSystem: WASD movement {(enableWASD ? "enabled" : "disabled")}");
        }
        
        /// <summary>
        /// Toggle mouse panning
        /// </summary>
        public void ToggleMousePan()
        {
            enableMousePan = !enableMousePan;
            Debug.Log($"CameraZoomSystem: Mouse panning {(enableMousePan ? "enabled" : "disabled")}");
        }
        
        /// <summary>
        /// Get current camera target position
        /// </summary>
        public Vector3 GetTargetPosition()
        {
            return targetPosition;
        }
        
        /// <summary>
        /// Get current camera target zoom
        /// </summary>
        public float GetTargetZoom()
        {
            return targetZoom;
        }
        
        /// <summary>
        /// Check if camera is currently following a target
        /// </summary>
        public bool IsFollowing()
        {
            return isFollowing;
        }
        
        /// <summary>
        /// Check if camera is currently panning
        /// </summary>
        public bool IsPanning()
        {
            return isPanning;
        }
        
        void OnDestroy()
        {
            // Clean up input actions
            if (moveAction != null) moveAction.Disable();
            if (zoomAction != null) zoomAction.Disable();
            if (panAction != null) panAction.Disable();
        }
    }
}

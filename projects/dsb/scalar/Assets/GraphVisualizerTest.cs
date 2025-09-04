using UnityEngine;

namespace Scalar
{
    /// <summary>
    /// Test script to debug GraphVisualizer branch node issues
    /// </summary>
    public class GraphVisualizerTest : MonoBehaviour
    {
        [Header("Test Controls")]
        [SerializeField] private bool enableDebugLogging = true;
        [SerializeField] private KeyCode refreshKey = KeyCode.R;
        [SerializeField] private KeyCode diagnoseKey = KeyCode.D;
        [SerializeField] private KeyCode branchRefreshKey = KeyCode.B;
        [SerializeField] private KeyCode logStateKey = KeyCode.L;
        
        private GraphVisualizer graphVisualizer;
        private GraphSystem graphSystem;
        private PartyVisualizer partyVisualizer;
        
        void Start()
        {
            // Find required components
            graphVisualizer = FindFirstObjectByType<GraphVisualizer>();
            graphSystem = FindFirstObjectByType<GraphSystem>();
            partyVisualizer = FindFirstObjectByType<PartyVisualizer>();
            
            if (graphVisualizer == null)
            {
                Debug.LogError("GraphVisualizerTest: No GraphVisualizer found!");
            }
            
            if (graphSystem == null)
            {
                Debug.LogError("GraphVisualizerTest: No GraphSystem found!");
            }
            
            if (partyVisualizer == null)
            {
                Debug.LogError("GraphVisualizerTest: No PartyVisualizer found!");
            }
            
            Debug.Log("GraphVisualizerTest: Test script initialized. Use keys to test:");
            Debug.Log("  R - Refresh visualization");
            Debug.Log("  D - Diagnose and fix issues");
            Debug.Log("  B - Force refresh branch nodes");
            Debug.Log("  L - Log current state");
        }
        
        void Update()
        {
            if (!enableDebugLogging) return;
            
            // Test key controls
            if (Input.GetKeyDown(refreshKey))
            {
                TestRefreshVisualization();
            }
            
            if (Input.GetKeyDown(diagnoseKey))
            {
                TestDiagnoseVisualization();
            }
            
            if (Input.GetKeyDown(branchRefreshKey))
            {
                TestRefreshBranchNodes();
            }
            
            if (Input.GetKeyDown(logStateKey))
            {
                TestLogState();
            }
        }
        
        /// <summary>
        /// Test refreshing the entire visualization
        /// </summary>
        private void TestRefreshVisualization()
        {
            Debug.Log("=== Testing Visualization Refresh ===");
            
            if (graphVisualizer != null)
            {
                graphVisualizer.ForceRefreshVisualization();
                Debug.Log("GraphVisualizerTest: Triggered visualization refresh");
            }
            else
            {
                Debug.LogError("GraphVisualizerTest: GraphVisualizer is null!");
            }
        }
        
        /// <summary>
        /// Test diagnosing and fixing visualization issues
        /// </summary>
        private void TestDiagnoseVisualization()
        {
            Debug.Log("=== Testing Visualization Diagnosis ===");
            
            if (graphVisualizer != null)
            {
                graphVisualizer.DiagnoseAndFixVisualization();
                Debug.Log("GraphVisualizerTest: Completed visualization diagnosis");
            }
            else
            {
                Debug.LogError("GraphVisualizerTest: GraphVisualizer is null!");
            }
        }
        
        /// <summary>
        /// Test refreshing branch nodes specifically
        /// </summary>
        private void TestRefreshBranchNodes()
        {
            Debug.Log("=== Testing Branch Node Refresh ===");
            
            if (graphVisualizer != null)
            {
                graphVisualizer.ForceRefreshBranchNodes();
                Debug.Log("GraphVisualizerTest: Completed branch node refresh");
            }
            else
            {
                Debug.LogError("GraphVisualizerTest: GraphVisualizer is null!");
            }
        }
        
        /// <summary>
        /// Test logging the current state
        /// </summary>
        private void TestLogState()
        {
            Debug.Log("=== Testing State Logging ===");
            
            if (graphVisualizer != null)
            {
                graphVisualizer.DebugVisualizerState();
                Debug.Log("GraphVisualizerTest: Completed state logging");
            }
            else
            {
                Debug.LogError("GraphVisualizerTest: GraphVisualizer is null!");
            }
            
            // Also log GraphSystem state
            if (graphSystem != null)
            {
                Debug.Log($"GraphSystem: Total nodes={graphSystem.GetTotalNodeCount()}, Visible nodes={graphSystem.GetVisibleNodes().Count}, Generated={graphSystem.IsGraphGenerated()}");
            }
            
            // Log PartyVisualizer state
            if (partyVisualizer != null)
            {
                Debug.Log("PartyVisualizer: State logged");
            }
        }
        
        /// <summary>
        /// Public method to test from other scripts
        /// </summary>
        public void TestAll()
        {
            Debug.Log("GraphVisualizerTest: Running all tests...");
            TestRefreshVisualization();
            TestDiagnoseVisualization();
            TestRefreshBranchNodes();
            TestLogState();
        }
    }
}

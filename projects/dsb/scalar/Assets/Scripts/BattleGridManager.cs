using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;

public class BattleGridManager : MonoBehaviour
{
    [Header("격자 설정")]
    public bool showGrid = true;
    public float gridSize = 1f;
    public int gridWidth = 10;
    public int gridHeight = 10;
    public Color gridColor = new Color(1f, 1f, 1f, 0.3f); // 반투명 흰색
    public float gridLineWidth = 0.05f;
    
    [Header("격자 시각화")]
    public Material gridMaterial;
    public GameObject gridLinePrefab;
    
    [Header("격자 위치")]
    public Vector3 gridCenter = Vector3.zero;
    public bool centerGrid = true;
    
    private List<GameObject> gridLines = new List<GameObject>();
    private bool isGridCreated = false;
    
    private void Start()
    {
        if (showGrid)
        {
            CreateGrid();
        }
    }
    
    private void Update()
    {
        // G 키로 격자 토글 (새로운 Input System 사용)
        if (Keyboard.current != null && Keyboard.current.gKey.wasPressedThisFrame)
        {
            ToggleGrid();
        }
    }
    
    public void CreateGrid()
    {
        if (isGridCreated)
        {
            ClearGrid();
        }
        
        // 단일 큰 쿼드로 격자 생성 (최적화)
        CreateOptimizedGrid();
        
        isGridCreated = true;
        Debug.Log($"격자 생성 완료: {gridWidth}x{gridHeight}, 크기: {gridSize}");
    }
    
    private void CreateOptimizedGrid()
    {
        Vector3 startPos = CalculateGridStartPosition();
        Vector3 centerPos = startPos + new Vector3(gridWidth * gridSize / 2f, gridHeight * gridSize / 2f, 0);
        
        // 단일 큰 쿼드로 격자 배경 생성
        GameObject gridBackground = GameObject.CreatePrimitive(PrimitiveType.Quad);
        gridBackground.name = "GridBackground";
        gridBackground.transform.parent = transform;
        gridBackground.transform.position = centerPos;
        gridBackground.transform.localScale = new Vector3(gridWidth * gridSize, gridHeight * gridSize, 1);
        
        // 배경 머티리얼 설정
        Renderer renderer = gridBackground.GetComponent<Renderer>();
        Material mat = gridMaterial != null ? gridMaterial : CreateDefaultGridMaterial();
        mat.color = new Color(gridColor.r, gridColor.g, gridColor.b, 0.05f); // 매우 투명한 배경
        renderer.material = mat;
        
        // 콜라이더 제거
        Collider collider = gridBackground.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        gridLines.Add(gridBackground);
        
        // 격자 선만 생성 (선의 개수 최소화)
        CreateGridLines(startPos);
    }
    
    private void CreateGridLines(Vector3 startPos)
    {
        // 세로선 생성 (gridWidth + 1개)
        for (int x = 0; x <= gridWidth; x++)
        {
            Vector3 linePos = startPos + new Vector3(x * gridSize, gridHeight * gridSize / 2f, 0);
            CreateGridLine(linePos, new Vector3(gridLineWidth, gridHeight * gridSize, 1), $"VerticalLine_{x}");
        }
        
        // 가로선 생성 (gridHeight + 1개)
        for (int y = 0; y <= gridHeight; y++)
        {
            Vector3 linePos = startPos + new Vector3(gridWidth * gridSize / 2f, y * gridSize, 0);
            CreateGridLine(linePos, new Vector3(gridWidth * gridSize, gridLineWidth, 1), $"HorizontalLine_{y}");
        }
    }
    
    private void CreateGridLine(Vector3 position, Vector3 scale, string name)
    {
        GameObject lineQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        lineQuad.name = name;
        lineQuad.transform.parent = transform;
        lineQuad.transform.position = position;
        lineQuad.transform.localScale = scale;
        
        // 머티리얼 설정
        Renderer renderer = lineQuad.GetComponent<Renderer>();
        Material mat = gridMaterial != null ? gridMaterial : CreateDefaultGridMaterial();
        mat.color = gridColor;
        renderer.material = mat;
        
        // 콜라이더 제거
        Collider collider = lineQuad.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        gridLines.Add(lineQuad);
    }
    
    private void CreateGridCell(Vector3 startPos, int x, int z)
    {
        Vector3 cellPosition = startPos + new Vector3(x * gridSize + gridSize / 2f, z * gridSize + gridSize / 2f, 0);
        
        // 2D 격자 셀을 위한 쿼드 생성
        GameObject cellQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        cellQuad.name = $"GridCell_{x}_{z}";
        cellQuad.transform.parent = transform;
        cellQuad.transform.position = cellPosition;
        cellQuad.transform.localScale = new Vector3(gridSize * 0.9f, gridSize * 0.9f, 1);
        
        // 2D용 머티리얼 설정
        Renderer renderer = cellQuad.GetComponent<Renderer>();
        Material mat = gridMaterial != null ? gridMaterial : CreateDefaultGridMaterial();
        mat.color = gridColor;
        renderer.material = mat;
        
        // 콜라이더 제거 (시각적 표현만 필요)
        Collider collider = cellQuad.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        gridLines.Add(cellQuad);
    }
    
    private void CreateGridLineCube(Vector3 position, Vector3 scale, string name)
    {
        GameObject lineCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        lineCube.name = name;
        lineCube.transform.parent = transform;
        lineCube.transform.position = position;
        lineCube.transform.localScale = scale;
        
        // 머티리얼 설정
        Renderer renderer = lineCube.GetComponent<Renderer>();
        Material mat = gridMaterial != null ? gridMaterial : CreateDefaultGridMaterial();
        mat.color = gridColor;
        renderer.material = mat;
        
        // 콜라이더 제거 (시각적 표현만 필요)
        Collider collider = lineCube.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        gridLines.Add(lineCube);
    }
    
    
    private Material CreateDefaultGridMaterial()
    {
        // 2D용 스프라이트 머티리얼 생성 (반투명 지원)
        Material mat = new Material(Shader.Find("Sprites/Default"));
        mat.color = gridColor;
        
        // 반투명 설정
        if (gridColor.a < 1.0f)
        {
            mat.SetFloat("_Mode", 3); // Transparent
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 1000; // 격자를 뒤로 보내기 위해 낮은 값 사용
        }
        
        return mat;
    }
    
    public Vector3 CalculateGridStartPosition()
    {
        if (centerGrid)
        {
            float halfWidth = (gridWidth * gridSize) / 2f;
            float halfHeight = (gridHeight * gridSize) / 2f;
            return gridCenter + new Vector3(-halfWidth, -halfHeight, 0);
        }
        else
        {
            return gridCenter;
        }
    }
    
    public void ClearGrid()
    {
        foreach (GameObject line in gridLines)
        {
            if (line != null)
            {
                DestroyImmediate(line);
            }
        }
        gridLines.Clear();
        isGridCreated = false;
        Debug.Log("격자 제거 완료");
    }
    
    public void ToggleGrid()
    {
        showGrid = !showGrid;
        if (showGrid)
        {
            CreateGrid();
        }
        else
        {
            ClearGrid();
        }
        Debug.Log($"격자 표시: {showGrid}");
    }
    
    public Vector3 GetGridPosition(Vector3 worldPosition)
    {
        Vector3 startPos = CalculateGridStartPosition();
        Vector3 relativePos = worldPosition - startPos;
        
        int gridX = Mathf.RoundToInt(relativePos.x / gridSize);
        int gridY = Mathf.RoundToInt(relativePos.y / gridSize);
        
        // 격자 범위 내로 제한
        gridX = Mathf.Clamp(gridX, 0, gridWidth - 1);
        gridY = Mathf.Clamp(gridY, 0, gridHeight - 1);
        
        return startPos + new Vector3(gridX * gridSize, gridY * gridSize, 0);
    }
    
    public Vector3 GetWorldPosition(int gridX, int gridY)
    {
        Vector3 startPos = CalculateGridStartPosition();
        return startPos + new Vector3(gridX * gridSize, gridY * gridSize, 0);
    }
    
    public bool IsValidGridPosition(int gridX, int gridY)
    {
        return gridX >= 0 && gridX < gridWidth && gridY >= 0 && gridY < gridHeight;
    }
    
    public void HighlightGridCell(int gridX, int gridY, Color highlightColor)
    {
        if (!IsValidGridPosition(gridX, gridY))
            return;
        
        Vector3 cellCenter = GetWorldPosition(gridX, gridY) + new Vector3(gridSize / 2f, gridSize / 2f, 0);
        
        // 하이라이트 쿼드 생성 (2D)
        GameObject highlight = GameObject.CreatePrimitive(PrimitiveType.Quad);
        highlight.name = $"GridHighlight_{gridX}_{gridY}";
        highlight.transform.position = cellCenter;
        highlight.transform.localScale = new Vector3(gridSize * 0.9f, gridSize * 0.9f, 1);
        
        // 하이라이트 머티리얼 설정
        Renderer renderer = highlight.GetComponent<Renderer>();
        Material highlightMat = new Material(Shader.Find("Sprites/Default"));
        highlightMat.color = highlightColor;
        renderer.material = highlightMat;
        
        // 콜라이더 제거
        Collider collider = highlight.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        // 2초 후 자동 제거
        Destroy(highlight, 2f);
    }
    
    public void UpdateGridSettings(float newGridSize, int newWidth, int newHeight)
    {
        gridSize = newGridSize;
        gridWidth = newWidth;
        gridHeight = newHeight;
        
        if (isGridCreated)
        {
            CreateGrid();
        }
    }
    
    private void OnDrawGizmos()
    {
        if (!showGrid) return;
        
        Vector3 startPos = CalculateGridStartPosition();
        
        Gizmos.color = gridColor;
        
        // 세로선 그리기 (2D)
        for (int x = 0; x <= gridWidth; x++)
        {
            Vector3 start = startPos + new Vector3(x * gridSize, 0, 0);
            Vector3 end = start + new Vector3(0, gridHeight * gridSize, 0);
            Gizmos.DrawLine(start, end);
        }
        
        // 가로선 그리기 (2D)
        for (int y = 0; y <= gridHeight; y++)
        {
            Vector3 start = startPos + new Vector3(0, y * gridSize, 0);
            Vector3 end = start + new Vector3(gridWidth * gridSize, 0, 0);
            Gizmos.DrawLine(start, end);
        }
    }
    
    [ContextMenu("격자 생성")]
    public void CreateGridFromContextMenu()
    {
        CreateGrid();
    }
    
    [ContextMenu("격자 제거")]
    public void ClearGridFromContextMenu()
    {
        ClearGrid();
    }
    
    [ContextMenu("격자 토글")]
    public void ToggleGridFromContextMenu()
    {
        ToggleGrid();
    }
}

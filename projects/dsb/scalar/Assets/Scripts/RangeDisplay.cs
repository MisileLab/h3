using UnityEngine;
using System.Collections.Generic;

public class RangeDisplay : MonoBehaviour
{
    [Header("Range Display Settings")]
    public Material attackRangeMaterial;
    public Material moveRangeMaterial;
    public Material skillRangeMaterial;
    
    [Header("Range Colors")]
    public Color attackRangeColor = new Color(1f, 0.3f, 0.3f, 0.5f); // 빨간색
    public Color moveRangeColor = new Color(0.3f, 0.3f, 1f, 0.5f);   // 파란색
    public Color skillRangeColor = new Color(0.3f, 1f, 0.3f, 0.5f);  // 초록색
    public Color targetHighlightColor = new Color(1f, 1f, 0f, 0.7f); // 노란색
    
    private List<GameObject> rangeIndicators = new List<GameObject>();
    private List<GameObject> targetHighlights = new List<GameObject>();
    private BattleGridManager gridManager;
    
    private void Start()
    {
        gridManager = FindObjectOfType<BattleGridManager>();
        if (gridManager == null)
        {
            Debug.LogError("BattleGridManager를 찾을 수 없습니다!");
        }
        
        // 기본 머티리얼이 없으면 생성
        CreateDefaultMaterials();
    }
    
    private void CreateDefaultMaterials()
    {
        if (attackRangeMaterial == null)
        {
            attackRangeMaterial = CreateRangeMaterial(attackRangeColor);
        }
        if (moveRangeMaterial == null)
        {
            moveRangeMaterial = CreateRangeMaterial(moveRangeColor);
        }
        if (skillRangeMaterial == null)
        {
            skillRangeMaterial = CreateRangeMaterial(skillRangeColor);
        }
    }
    
    private Material CreateRangeMaterial(Color color)
    {
        Material mat = new Material(Shader.Find("Sprites/Default"));
        mat.color = color;
        
        // 투명 설정
        mat.SetFloat("_Mode", 3); // Transparent
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 1500; // 격자보다 앞에, 캐릭터보다 뒤에
        
        return mat;
    }
    
    /// <summary>
    /// 공격 사정거리를 표시합니다
    /// </summary>
    public void ShowAttackRange(Vector3 centerPosition, int range)
    {
        ShowRange(centerPosition, range, attackRangeMaterial, RangeType.Attack);
        Debug.Log($"공격 사정거리 표시: 중심({centerPosition}), 범위({range})");
    }
    
    /// <summary>
    /// 이동 범위를 표시합니다
    /// </summary>
    public void ShowMoveRange(Vector3 centerPosition, int range)
    {
        ShowRange(centerPosition, range, moveRangeMaterial, RangeType.Move);
        Debug.Log($"이동 범위 표시: 중심({centerPosition}), 범위({range})");
    }
    
    /// <summary>
    /// 스킬 범위를 표시합니다
    /// </summary>
    public void ShowSkillRange(Vector3 centerPosition, int range)
    {
        ShowRange(centerPosition, range, skillRangeMaterial, RangeType.Skill);
        Debug.Log($"스킬 범위 표시: 중심({centerPosition}), 범위({range})");
    }
    
    public enum RangeType
    {
        Attack,
        Move,
        Skill
    }
    
    private void ShowRange(Vector3 centerWorldPosition, int range, Material material, RangeType rangeType)
    {
        if (gridManager == null) return;
        
        ClearRangeDisplay();
        
        // 월드 좌표를 격자 좌표로 변환
        Vector3 gridPos = gridManager.GetGridPosition(centerWorldPosition);
        Vector3 startPos = gridManager.CalculateGridStartPosition();
        
        // 격자 좌표 계산
        Vector3 relativePos = gridPos - startPos;
        int centerGridX = Mathf.RoundToInt(relativePos.x / gridManager.gridSize);
        int centerGridY = Mathf.RoundToInt(relativePos.y / gridManager.gridSize);
        
        // 맨하탄 거리로 범위 내의 격자 셀들 찾기
        for (int dx = -range; dx <= range; dx++)
        {
            for (int dy = -range; dy <= range; dy++)
            {
                // 맨하탄 거리 체크
                if (Mathf.Abs(dx) + Mathf.Abs(dy) <= range)
                {
                    int gridX = centerGridX + dx;
                    int gridY = centerGridY + dy;
                    
                    // 격자 범위 내인지 확인
                    if (gridManager.IsValidGridPosition(gridX, gridY))
                    {
                        CreateRangeIndicator(gridX, gridY, material, rangeType);
                    }
                }
            }
        }
        
        Debug.Log($"범위 표시 생성 완료: {rangeIndicators.Count}개 셀");
    }
    
    private void CreateRangeIndicator(int gridX, int gridY, Material material, RangeType rangeType)
    {
        Vector3 worldPos = gridManager.GetWorldPosition(gridX, gridY);
        Vector3 cellCenter = worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
        
        // 범위 표시용 쿼드 생성
        GameObject indicator = GameObject.CreatePrimitive(PrimitiveType.Quad);
        indicator.name = $"RangeIndicator_{rangeType}_{gridX}_{gridY}";
        indicator.transform.position = cellCenter + new Vector3(0, 0, -0.1f); // 살짝 앞으로
        indicator.transform.localScale = new Vector3(gridManager.gridSize * 0.9f, gridManager.gridSize * 0.9f, 1);
        
        // 머티리얼 적용
        Renderer renderer = indicator.GetComponent<Renderer>();
        renderer.material = material;
        
        // 콜라이더 제거
        Collider collider = indicator.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        // 범위 표시기에 그리드 좌표 정보 추가
        RangeIndicatorData data = indicator.AddComponent<RangeIndicatorData>();
        data.gridX = gridX;
        data.gridY = gridY;
        data.rangeType = rangeType;
        
        rangeIndicators.Add(indicator);
    }
    
    /// <summary>
    /// 특정 위치를 타겟으로 하이라이트합니다
    /// </summary>
    public void HighlightTarget(int gridX, int gridY)
    {
        ClearTargetHighlights();
        
        if (!gridManager.IsValidGridPosition(gridX, gridY)) return;
        
        Vector3 worldPos = gridManager.GetWorldPosition(gridX, gridY);
        Vector3 cellCenter = worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
        
        // 타겟 하이라이트용 쿼드 생성
        GameObject highlight = GameObject.CreatePrimitive(PrimitiveType.Quad);
        highlight.name = $"TargetHighlight_{gridX}_{gridY}";
        highlight.transform.position = cellCenter + new Vector3(0, 0, -0.2f); // 범위 표시보다 앞으로
        highlight.transform.localScale = new Vector3(gridManager.gridSize * 0.95f, gridManager.gridSize * 0.95f, 1);
        
        // 하이라이트 머티리얼 적용
        Renderer renderer = highlight.GetComponent<Renderer>();
        Material highlightMat = CreateRangeMaterial(targetHighlightColor);
        renderer.material = highlightMat;
        
        // 콜라이더 제거
        Collider collider = highlight.GetComponent<Collider>();
        if (collider != null)
        {
            DestroyImmediate(collider);
        }
        
        targetHighlights.Add(highlight);
    }
    
    /// <summary>
    /// 범위 표시를 모두 제거합니다
    /// </summary>
    public void ClearRangeDisplay()
    {
        foreach (GameObject indicator in rangeIndicators)
        {
            if (indicator != null)
            {
                DestroyImmediate(indicator);
            }
        }
        rangeIndicators.Clear();
    }
    
    /// <summary>
    /// 타겟 하이라이트를 모두 제거합니다
    /// </summary>
    public void ClearTargetHighlights()
    {
        foreach (GameObject highlight in targetHighlights)
        {
            if (highlight != null)
            {
                DestroyImmediate(highlight);
            }
        }
        targetHighlights.Clear();
    }
    
    /// <summary>
    /// 모든 표시를 제거합니다
    /// </summary>
    public void ClearAll()
    {
        ClearRangeDisplay();
        ClearTargetHighlights();
    }
    
    /// <summary>
    /// 특정 격자 위치가 현재 표시된 범위 내에 있는지 확인합니다
    /// </summary>
    public bool IsInDisplayedRange(int gridX, int gridY)
    {
        foreach (GameObject indicator in rangeIndicators)
        {
            RangeIndicatorData data = indicator.GetComponent<RangeIndicatorData>();
            if (data != null && data.gridX == gridX && data.gridY == gridY)
            {
                return true;
            }
        }
        return false;
    }
    
    /// <summary>
    /// 현재 표시된 범위의 타입을 반환합니다
    /// </summary>
    public RangeType? GetCurrentRangeType()
    {
        if (rangeIndicators.Count > 0)
        {
            RangeIndicatorData data = rangeIndicators[0].GetComponent<RangeIndicatorData>();
            return data?.rangeType;
        }
        return null;
    }
}

/// <summary>
/// 범위 표시기에 붙어있는 데이터 컴포넌트
/// </summary>
public class RangeIndicatorData : MonoBehaviour
{
    public int gridX;
    public int gridY;
    public RangeDisplay.RangeType rangeType;
}

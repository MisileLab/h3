using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using System;

public class TargetSelector : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera battleCamera;
    
    [Header("Debug Settings")]
    public bool debugMode = false;
    
    private BattleGridManager gridManager;
    private RangeDisplay rangeDisplay;
    
    // 타겟 선택 이벤트
    public static event Action<int, int, RangeDisplay.RangeType> OnTargetSelected;
    public static event Action<Vector3> OnTargetPreview; // 마우스 오버 시
    
    // 현재 선택 모드 상태
    public bool isSelectionActive = false;
    public RangeDisplay.RangeType currentSelectionType;
    
    private void Start()
    {
        gridManager = FindObjectOfType<BattleGridManager>();
        rangeDisplay = FindObjectOfType<RangeDisplay>();
        
        if (battleCamera == null)
        {
            battleCamera = Camera.main;
        }
        
        if (gridManager == null)
        {
            Debug.LogError("BattleGridManager를 찾을 수 없습니다!");
        }
        
        if (rangeDisplay == null)
        {
            Debug.LogError("RangeDisplay를 찾을 수 없습니다!");
        }
    }
    
    private void Update()
    {
        if (!isSelectionActive) return;
        
        // 마우스 위치에 따른 타겟 미리보기
        HandleTargetPreview();
        
        // 마우스 클릭 처리
        HandleMouseClick();
        
        // ESC 키로 선택 취소
        if (Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
        {
            CancelTargetSelection();
        }
    }
    
    private void HandleTargetPreview()
    {
        Vector3 mousePosition = GetMouseWorldPosition();
        if (mousePosition == Vector3.zero) return;
        
        Vector2Int gridCoords = GetGridCoordinates(mousePosition);
        
        // 유효한 격자 위치이고 범위 내에 있으면 하이라이트
        if (gridManager.IsValidGridPosition(gridCoords.x, gridCoords.y) && 
            rangeDisplay.IsInDisplayedRange(gridCoords.x, gridCoords.y))
        {
            rangeDisplay.HighlightTarget(gridCoords.x, gridCoords.y);
            OnTargetPreview?.Invoke(mousePosition);
        }
        else
        {
            rangeDisplay.ClearTargetHighlights();
        }
    }
    
    private void HandleMouseClick()
    {
        // 마우스 왼쪽 버튼 클릭 체크 (Input System 사용)
        if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame)
        {
            Vector3 mousePosition = GetMouseWorldPosition();
            if (mousePosition == Vector3.zero) return;
            
            Vector2Int gridCoords = GetGridCoordinates(mousePosition);
            
            if (debugMode)
            {
                Debug.Log($"마우스 클릭: 월드({mousePosition}) -> 격자({gridCoords.x}, {gridCoords.y})");
            }
            
            // 유효한 타겟인지 확인
            if (IsValidTarget(gridCoords.x, gridCoords.y))
            {
                SelectTarget(gridCoords.x, gridCoords.y);
            }
            else
            {
                if (debugMode)
                {
                    Debug.Log($"잘못된 타겟: ({gridCoords.x}, {gridCoords.y})");
                }
            }
        }
    }
    
    private Vector3 GetMouseWorldPosition()
    {
        if (battleCamera == null) return Vector3.zero;
        
        Vector3 mouseScreenPos = Mouse.current.position.ReadValue();
        mouseScreenPos.z = -battleCamera.transform.position.z; // 카메라 거리만큼 z값 설정
        
        Vector3 worldPos = battleCamera.ScreenToWorldPoint(mouseScreenPos);
        worldPos.z = 0; // 2D이므로 z를 0으로 설정
        
        return worldPos;
    }
    
    private Vector2Int GetGridCoordinates(Vector3 worldPosition)
    {
        if (gridManager == null) return Vector2Int.zero;
        
        Vector3 startPos = gridManager.CalculateGridStartPosition();
        Vector3 relativePos = worldPosition - startPos;
        
        int gridX = Mathf.FloorToInt(relativePos.x / gridManager.gridSize);
        int gridY = Mathf.FloorToInt(relativePos.y / gridManager.gridSize);
        
        return new Vector2Int(gridX, gridY);
    }
    
    private bool IsValidTarget(int gridX, int gridY)
    {
        // 격자 범위 내인지 확인
        if (!gridManager.IsValidGridPosition(gridX, gridY))
            return false;
        
        // 표시된 범위 내인지 확인
        if (!rangeDisplay.IsInDisplayedRange(gridX, gridY))
            return false;
        
        // 타겟 타입별 추가 검증
        switch (currentSelectionType)
        {
            case RangeDisplay.RangeType.Attack:
                return IsValidAttackTarget(gridX, gridY);
            case RangeDisplay.RangeType.Move:
                return IsValidMoveTarget(gridX, gridY);
            case RangeDisplay.RangeType.Skill:
                return IsValidSkillTarget(gridX, gridY);
        }
        
        return true;
    }
    
    private bool IsValidAttackTarget(int gridX, int gridY)
    {
        // 공격 타겟 검증 - 적이 있는 위치인지 확인
        Vector3 worldPos = gridManager.GetWorldPosition(gridX, gridY);
        Vector3 centerPos = worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
        
        // 해당 위치에 적이 있는지 확인
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive && Vector3.Distance(enemy.transform.position, centerPos) < gridManager.gridSize * 0.6f)
            {
                return true;
            }
        }
        
        return false; // 공격 타겟은 적이 있어야만 유효
    }
    
    private bool IsValidMoveTarget(int gridX, int gridY)
    {
        // 이동 타겟 검증 - 다른 캐릭터가 없는 빈 공간인지 확인
        Vector3 worldPos = gridManager.GetWorldPosition(gridX, gridY);
        Vector3 centerPos = worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
        
        // 해당 위치에 다른 캐릭터가 있는지 확인
        // 아군 확인
        MechCharacter[] mechs = FindObjectsOfType<MechCharacter>();
        foreach (MechCharacter mech in mechs)
        {
            if (mech.isAlive && Vector3.Distance(mech.transform.position, centerPos) < gridManager.gridSize * 0.6f)
            {
                return false; // 아군이 있으면 이동 불가
            }
        }
        
        // 적 확인
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive && Vector3.Distance(enemy.transform.position, centerPos) < gridManager.gridSize * 0.6f)
            {
                return false; // 적이 있으면 이동 불가
            }
        }
        
        return true; // 빈 공간이면 이동 가능
    }
    
    private bool IsValidSkillTarget(int gridX, int gridY)
    {
        // 스킬 타겟 검증 - 스킬 타입에 따라 다르게 처리할 수 있음
        // 현재는 범위 내면 모두 유효하다고 처리
        return true;
    }
    
    private void SelectTarget(int gridX, int gridY)
    {
        if (debugMode)
        {
            Debug.Log($"타겟 선택됨: ({gridX}, {gridY}), 타입: {currentSelectionType}");
        }
        
        // 타겟 선택 이벤트 발생
        OnTargetSelected?.Invoke(gridX, gridY, currentSelectionType);
        
        // 선택 모드 종료
        EndTargetSelection();
    }
    
    /// <summary>
    /// 타겟 선택 모드를 시작합니다
    /// </summary>
    public void StartTargetSelection(RangeDisplay.RangeType selectionType)
    {
        isSelectionActive = true;
        currentSelectionType = selectionType;
        
        if (debugMode)
        {
            Debug.Log($"타겟 선택 모드 시작: {selectionType}");
        }
    }
    
    /// <summary>
    /// 타겟 선택 모드를 종료합니다
    /// </summary>
    public void EndTargetSelection()
    {
        isSelectionActive = false;
        
        if (rangeDisplay != null)
        {
            rangeDisplay.ClearAll();
        }
        
        if (debugMode)
        {
            Debug.Log("타겟 선택 모드 종료");
        }
    }
    
    /// <summary>
    /// 타겟 선택을 취소합니다
    /// </summary>
    public void CancelTargetSelection()
    {
        if (debugMode)
        {
            Debug.Log("타겟 선택 취소됨");
        }
        
        EndTargetSelection();
    }
    
    /// <summary>
    /// 특정 월드 위치의 격자 좌표를 반환합니다 (외부에서 사용 가능)
    /// </summary>
    public Vector2Int WorldToGridPosition(Vector3 worldPosition)
    {
        return GetGridCoordinates(worldPosition);
    }
    
    /// <summary>
    /// 특정 격자 좌표의 월드 위치를 반환합니다 (외부에서 사용 가능)
    /// </summary>
    public Vector3 GridToWorldPosition(int gridX, int gridY)
    {
        if (gridManager == null) return Vector3.zero;
        
        Vector3 worldPos = gridManager.GetWorldPosition(gridX, gridY);
        return worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
    }
}

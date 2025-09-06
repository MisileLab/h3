using UnityEngine;

/// <summary>
/// 행동 포인트(AP) 시스템을 관리하는 클래스
/// 모든 유닛은 턴마다 정해진 AP를 가지며, 행동 시 AP를 소모합니다.
/// </summary>
[System.Serializable]
public class ActionPoint
{
    [Header("AP 설정")]
    public int maxAP = 3;           // 최대 AP
    public int currentAP = 3;       // 현재 AP
    
    public ActionPoint(int maxActionPoints = 3)
    {
        maxAP = maxActionPoints;
        currentAP = maxActionPoints;
    }
    
    /// <summary>
    /// 턴 시작 시 AP를 최대값으로 복구
    /// </summary>
    public void RefreshAP()
    {
        currentAP = maxAP;
        Debug.Log($"AP가 {maxAP}로 복구되었습니다.");
    }
    
    /// <summary>
    /// AP를 소모합니다.
    /// </summary>
    /// <param name="amount">소모할 AP량</param>
    /// <returns>성공적으로 소모했으면 true</returns>
    public bool ConsumeAP(int amount)
    {
        if (currentAP >= amount)
        {
            currentAP -= amount;
            Debug.Log($"AP {amount} 소모. 남은 AP: {currentAP}");
            return true;
        }
        
        Debug.LogWarning($"AP가 부족합니다. 필요: {amount}, 보유: {currentAP}");
        return false;
    }
    
    /// <summary>
    /// 지정된 AP량을 사용할 수 있는지 확인
    /// </summary>
    /// <param name="amount">확인할 AP량</param>
    /// <returns>사용 가능하면 true</returns>
    public bool CanUseAP(int amount)
    {
        return currentAP >= amount;
    }
    
    /// <summary>
    /// AP를 회복합니다 (최대치 초과 불가)
    /// </summary>
    /// <param name="amount">회복할 AP량</param>
    public void RecoverAP(int amount)
    {
        currentAP = Mathf.Min(maxAP, currentAP + amount);
        Debug.Log($"AP {amount} 회복. 현재 AP: {currentAP}");
    }
    
    /// <summary>
    /// 최대 AP를 증가시킵니다 (버프 등)
    /// </summary>
    /// <param name="amount">증가할 최대 AP량</param>
    public void IncreaseMaxAP(int amount)
    {
        maxAP += amount;
        Debug.Log($"최대 AP가 {amount} 증가했습니다. 새로운 최대 AP: {maxAP}");
    }
    
    /// <summary>
    /// 현재 AP 비율 (0~1)
    /// </summary>
    public float GetAPRatio()
    {
        return maxAP > 0 ? (float)currentAP / maxAP : 0f;
    }
}

/// <summary>
/// 행동별 AP 소모량을 정의하는 열거형
/// </summary>
public enum ActionCost
{
    Move = 1,           // 이동
    Attack = 1,         // 기본 공격
    Defense = 1,        // 방어
    Watch = 2,          // 경계
    SpecialAbility = 1, // 특수 능력 (기본값, 개별 설정 가능)
    UseItem = 1,        // 아이템 사용
    CooperativeAction = 2 // 협력 행동 (2명이 함께 사용하므로 높은 비용)
}

using UnityEngine;
using System;

/// <summary>
/// 기계의 부위별 손상 시스템을 관리하는 클래스
/// 각 부위는 전체 HP의 일정 비율을 차지하며, 파괴 시 특별한 패널티가 발생합니다.
/// </summary>
[System.Serializable]
public class BodyPart
{
    [Header("부위 정보")]
    public BodyPartType partType;
    public string partName;
    
    [Header("HP 설정")]
    public float hpPercentage;      // 전체 HP에서 차지하는 비율 (0~1)
    public int maxHP;               // 이 부위의 최대 HP
    public int currentHP;           // 현재 HP
    
    [Header("상태")]
    public bool isDestroyed = false;
    public DamageLevel damageLevel = DamageLevel.None;
    
    // 이벤트
    public static event Action<BodyPart, DamageLevel> OnDamageLevelChanged;
    public static event Action<BodyPart> OnPartDestroyed;
    
    public BodyPart(BodyPartType type, float hpPercent, int totalMaxHP)
    {
        partType = type;
        partName = GetPartName(type);
        hpPercentage = hpPercent;
        maxHP = Mathf.RoundToInt(totalMaxHP * hpPercent);
        currentHP = maxHP;
        isDestroyed = false;
        damageLevel = DamageLevel.None;
    }
    
    /// <summary>
    /// 부위에 피해를 가합니다
    /// </summary>
    /// <param name="damage">가할 피해량</param>
    /// <returns>실제로 입은 피해량</returns>
    public int TakeDamage(int damage)
    {
        if (isDestroyed) return 0;
        
        int actualDamage = Mathf.Min(damage, currentHP);
        currentHP -= actualDamage;
        
        // 손상 단계 업데이트
        DamageLevel newDamageLevel = CalculateDamageLevel();
        if (newDamageLevel != damageLevel)
        {
            damageLevel = newDamageLevel;
            OnDamageLevelChanged?.Invoke(this, damageLevel);
            Debug.Log($"{partName} 손상 단계: {damageLevel}");
        }
        
        // 부위 파괴 확인
        if (currentHP <= 0 && !isDestroyed)
        {
            DestroyPart();
        }
        
        return actualDamage;
    }
    
    /// <summary>
    /// 부위를 치료합니다
    /// </summary>
    /// <param name="healAmount">치료량</param>
    /// <param name="canRepairDestroyed">파괴된 부위도 수리 가능한지</param>
    /// <returns>실제 치료량</returns>
    public int Heal(int healAmount, bool canRepairDestroyed = false)
    {
        if (isDestroyed && !canRepairDestroyed) return 0;
        
        // 파괴된 부위를 수리하는 경우
        if (isDestroyed && canRepairDestroyed)
        {
            isDestroyed = false;
            currentHP = 1; // 최소 HP로 복구
            Debug.Log($"{partName}이 응급 수리되었습니다!");
        }
        
        int actualHeal = Mathf.Min(healAmount, maxHP - currentHP);
        currentHP += actualHeal;
        
        // 손상 단계 업데이트
        DamageLevel newDamageLevel = CalculateDamageLevel();
        if (newDamageLevel != damageLevel)
        {
            damageLevel = newDamageLevel;
            OnDamageLevelChanged?.Invoke(this, damageLevel);
            Debug.Log($"{partName} 손상 단계: {damageLevel}");
        }
        
        return actualHeal;
    }
    
    /// <summary>
    /// 부위를 파괴합니다
    /// </summary>
    private void DestroyPart()
    {
        isDestroyed = true;
        currentHP = 0;
        damageLevel = DamageLevel.Destroyed;
        OnPartDestroyed?.Invoke(this);
        Debug.Log($"<color=red>{partName}이 파괴되었습니다!</color>");
    }
    
    /// <summary>
    /// 현재 HP 비율을 기준으로 손상 단계를 계산합니다
    /// </summary>
    /// <returns>현재 손상 단계</returns>
    private DamageLevel CalculateDamageLevel()
    {
        if (isDestroyed) return DamageLevel.Destroyed;
        
        float hpRatio = (float)currentHP / maxHP;
        
        if (hpRatio >= 0.75f) return DamageLevel.None;
        if (hpRatio >= 0.50f) return DamageLevel.Minor;
        if (hpRatio >= 0.25f) return DamageLevel.Major;
        if (hpRatio > 0f) return DamageLevel.Critical;
        
        return DamageLevel.Destroyed;
    }
    
    /// <summary>
    /// 부위 타입에 따른 이름을 반환합니다
    /// </summary>
    /// <param name="type">부위 타입</param>
    /// <returns>부위 이름</returns>
    private string GetPartName(BodyPartType type)
    {
        return type switch
        {
            BodyPartType.Head => "머리/센서",
            BodyPartType.Torso => "몸통/코어",
            BodyPartType.RightArm => "오른팔/주무장",
            BodyPartType.LeftArm => "왼팔/보조장비",
            BodyPartType.Legs => "다리/이동부",
            _ => "알 수 없는 부위"
        };
    }
    
    /// <summary>
    /// 부위 파괴 시 적용되는 효과를 반환합니다
    /// </summary>
    /// <returns>파괴 효과 설명</returns>
    public BodyPartEffect GetDestroyEffect()
    {
        return partType switch
        {
            BodyPartType.Head => new BodyPartEffect
            {
                accuracyModifier = -50,
                evasionModifier = -50,
                canUseHacking = false,
                canUseScout = false,
                description = "명중률, 회피율 대폭 하락. 해킹 및 정찰 능력 사용 불가."
            },
            BodyPartType.Torso => new BodyPartEffect
            {
                isIncapacitated = true,
                description = "즉시 기능 정지 (전투 불능)"
            },
            BodyPartType.RightArm => new BodyPartEffect
            {
                attackPowerModifier = -50,
                canUseMainWeapon = false,
                description = "주 공격 스킬 사용 불가. 공격력 -50%."
            },
            BodyPartType.LeftArm => new BodyPartEffect
            {
                canUseShield = false,
                canUseRepairTool = false,
                description = "보조 능력(방패, 수리툴 등) 사용 불가."
            },
            BodyPartType.Legs => new BodyPartEffect
            {
                movementRange = 0,
                evasionModifier = -70,
                requiresCarrying = true,
                description = "이동 불가. 회피율 -70%. (운반 필요)"
            },
            _ => new BodyPartEffect { description = "알 수 없는 효과" }
        };
    }
    
    /// <summary>
    /// HP 비율 반환 (0~1)
    /// </summary>
    public float GetHPRatio()
    {
        return maxHP > 0 ? (float)currentHP / maxHP : 0f;
    }
    
    /// <summary>
    /// 부위 상태를 문자열로 반환
    /// </summary>
    public override string ToString()
    {
        return $"{partName}: {currentHP}/{maxHP} ({damageLevel})";
    }
}

/// <summary>
/// 기계의 부위 타입
/// </summary>
public enum BodyPartType
{
    Head,       // 머리/센서 (15%)
    Torso,      // 몸통/코어 (40%)
    RightArm,   // 오른팔/주무장 (20%)
    LeftArm,    // 왼팔/보조장비 (15%)
    Legs        // 다리/이동부 (10%)
}

/// <summary>
/// 손상 단계
/// </summary>
public enum DamageLevel
{
    None,       // 손상 없음 (75% 이상)
    Minor,      // 경미 손상 (50-74%)
    Major,      // 중간 손상 (25-49%)
    Critical,   // 심각 손상 (1-24%)
    Destroyed   // 완전 파괴 (0%)
}

/// <summary>
/// 부위 파괴 시 적용되는 효과
/// </summary>
[System.Serializable]
public struct BodyPartEffect
{
    public int attackPowerModifier;     // 공격력 수정치 (%)
    public int accuracyModifier;        // 명중률 수정치 (%)
    public int evasionModifier;         // 회피율 수정치 (%)
    public int movementRange;           // 이동 거리 (0이면 이동 불가)
    
    public bool isIncapacitated;        // 완전 전투 불능 여부
    public bool requiresCarrying;       // 운반이 필요한지
    public bool canUseMainWeapon;       // 주무기 사용 가능 여부
    public bool canUseShield;           // 방패 사용 가능 여부
    public bool canUseRepairTool;       // 수리 도구 사용 가능 여부
    public bool canUseHacking;          // 해킹 능력 사용 가능 여부
    public bool canUseScout;            // 정찰 능력 사용 가능 여부
    
    public string description;          // 효과 설명
}

using UnityEngine;

[System.Serializable]
public enum BodyPartType
{
    Head,       // 머리/센서 (15%)
    Torso,      // 몸통/코어 (40%)
    RightArm,   // 오른팔/주무장 (20%)
    LeftArm,    // 왼팔/보조장비 (15%)
    Legs        // 다리/이동부 (10%)
}

[System.Serializable]
public class MechBodyPart
{
    [Header("부위 정보")]
    public BodyPartType partType;
    public string partName;
    public float maxHP;
    public float currentHP;
    
    [Header("상태")]
    public bool isDestroyed = false;
    public bool isDamaged = false;
    
    [Header("파괴 시 효과")]
    public string destructionEffect;
    
    public MechBodyPart(BodyPartType type, string name, float hp)
    {
        partType = type;
        partName = name;
        maxHP = hp;
        currentHP = hp;
        isDestroyed = false;
        isDamaged = false;
    }
    
    public void TakeDamage(float damage)
    {
        currentHP = Mathf.Max(0, currentHP - damage);
        
        if (currentHP <= 0 && !isDestroyed)
        {
            isDestroyed = true;
            Debug.Log($"{partName} 부위가 파괴되었습니다!");
        }
        else if (currentHP < maxHP * 0.75f)
        {
            isDamaged = true;
        }
    }
    
    public void Repair(float amount)
    {
        currentHP = Mathf.Min(maxHP, currentHP + amount);
        if (currentHP >= maxHP * 0.75f)
        {
            isDamaged = false;
        }
        if (currentHP > 0)
        {
            isDestroyed = false;
        }
    }
    
    public float GetHPPercentage()
    {
        return currentHP / maxHP;
    }
    
    public DamageLevel GetDamageLevel()
    {
        float percentage = GetHPPercentage();
        
        if (percentage <= 0) return DamageLevel.Destroyed;
        if (percentage <= 0.25f) return DamageLevel.Critical;
        if (percentage <= 0.5f) return DamageLevel.Severe;
        if (percentage <= 0.75f) return DamageLevel.Moderate;
        return DamageLevel.Minor;
    }
}

public enum DamageLevel
{
    Minor,      // 경미 손상 (75% 이상)
    Moderate,   // 중간 손상 (50-74%)
    Severe,     // 심각 손상 (25-49%)
    Critical,   // 기능 정지 (0-24%)
    Destroyed   // 완전 파괴 (0%)
}

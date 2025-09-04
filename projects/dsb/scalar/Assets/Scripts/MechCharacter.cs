using UnityEngine;
using System.Collections.Generic;
using System;

[System.Serializable]
public enum MechType
{
    Rex,    // 프론트라인 가디언 (탱커)
    Luna,   // 테크니컬 서포터 (힐러/해커)
    Zero,   // 스피드 스카우트 (정찰/기동)
    Nova    // 헤비 어태커 (광역 딜러)
}

[System.Serializable]
public class MechStats
{
    [Header("기본 스탯")]
    public int maxHP = 100;
    public int currentHP = 100;
    public int maxAP = 3;
    public int currentAP = 3;
    public int speed = 10;
    public int attack = 20;
    public int defense = 15;
    public int accuracy = 80;
    public int evasion = 20;
    
    [Header("특수 능력")]
    public int hackSkill = 0;      // 해킹 능력
    public int repairSkill = 0;    // 수리 능력
    public int stealthSkill = 0;   // 은신 능력
    public int leadership = 0;     // 리더십 (협력 스킬 강화)
}

public class MechCharacter : MonoBehaviour
{
    [Header("기계 정보")]
    public MechType mechType;
    public string mechName;
    public MechStats stats;
    
    [Header("부위 시스템")]
    public List<MechBodyPart> bodyParts = new List<MechBodyPart>();
    
    [Header("상태")]
    public bool isAlive = true;
    public bool isInCombat = false;
    public bool isGuarding = false;
    public bool isInStealth = false;
    
    [Header("협력 시스템")]
    public Dictionary<MechType, int> trustLevels = new Dictionary<MechType, int>();
    public List<CooperationSkill> availableCoopSkills = new List<CooperationSkill>();
    
    [Header("쿨다운")]
    public Dictionary<string, float> skillCooldowns = new Dictionary<string, float>();
    
    // 이벤트
    public static event Action<MechCharacter, string> OnDialogueTriggered;
    public static event Action<MechCharacter, MechBodyPart, float> OnBodyPartDamaged;
    public static event Action<MechCharacter, MechBodyPart> OnBodyPartDestroyed;
    
    private void Start()
    {
        InitializeBodyParts();
        InitializeTrustLevels();
        InitializeCoopSkills();
    }
    
    protected void InitializeBodyParts()
    {
        // 기계 타입에 따른 부위 HP 설정
        float totalHP = stats.maxHP;
        
        bodyParts.Clear();
        bodyParts.Add(new MechBodyPart(BodyPartType.Head, "센서 시스템", totalHP * 0.15f));
        bodyParts.Add(new MechBodyPart(BodyPartType.Torso, "코어 시스템", totalHP * 0.40f));
        bodyParts.Add(new MechBodyPart(BodyPartType.RightArm, "주무장", totalHP * 0.20f));
        bodyParts.Add(new MechBodyPart(BodyPartType.LeftArm, "보조장비", totalHP * 0.15f));
        bodyParts.Add(new MechBodyPart(BodyPartType.Legs, "이동부", totalHP * 0.10f));
    }
    
    protected void InitializeTrustLevels()
    {
        foreach (MechType type in Enum.GetValues(typeof(MechType)))
        {
            if (type != mechType)
            {
                trustLevels[type] = 0; // 기본 신뢰도 0
            }
        }
    }
    
    protected void InitializeCoopSkills()
    {
        // 기본 협력 스킬들
        availableCoopSkills.Add(new CooperationSkill("가드", "동료를 보호합니다", 1, new List<MechType> { MechType.Rex }, CooperationType.Guard));
        availableCoopSkills.Add(new CooperationSkill("응급처치", "동료의 부위를 수리합니다", 2, new List<MechType> { MechType.Luna }, CooperationType.EmergencyRepair));
        availableCoopSkills.Add(new CooperationSkill("전술이동", "위치를 교환합니다", 1, new List<MechType> { MechType.Zero }, CooperationType.TacticalSwap));
        availableCoopSkills.Add(new CooperationSkill("연계공격", "함께 공격합니다", 2, new List<MechType> { MechType.Nova }, CooperationType.LinkAttack));
    }
    
    public virtual void TakeDamage(float damage, BodyPartType targetPart = BodyPartType.Torso)
    {
        if (!isAlive) return;
        
        MechBodyPart part = GetBodyPart(targetPart);
        if (part != null)
        {
            float actualDamage = CalculateActualDamage(damage);
            part.TakeDamage(actualDamage);
            
            OnBodyPartDamaged?.Invoke(this, part, actualDamage);
            
            if (part.isDestroyed)
            {
                OnBodyPartDestroyed?.Invoke(this, part);
                ApplyDestructionEffects(part);
            }
            
            // 몸통 파괴 시 기체 파괴
            if (targetPart == BodyPartType.Torso && part.isDestroyed)
            {
                DestroyMech();
            }
            
            TriggerDamageDialogue(part);
        }
    }
    
    private float CalculateActualDamage(float baseDamage)
    {
        float defenseModifier = 1.0f - (stats.defense / 100.0f);
        return baseDamage * defenseModifier;
    }
    
    private void ApplyDestructionEffects(MechBodyPart part)
    {
        switch (part.partType)
        {
            case BodyPartType.Head:
                stats.accuracy -= 30;
                stats.evasion -= 20;
                break;
            case BodyPartType.RightArm:
                stats.attack -= (int)(stats.attack * 0.5f);
                break;
            case BodyPartType.LeftArm:
                // 보조 능력 사용 불가
                break;
            case BodyPartType.Legs:
                stats.evasion -= (int)(stats.evasion * 0.7f);
                // 이동 불가 상태로 설정
                break;
        }
    }
    
    private void DestroyMech()
    {
        isAlive = false;
        TriggerDialogue("기체 파괴", "드라이브만이라도... 회수해줘...");
    }
    
    public void RepairBodyPart(BodyPartType partType, float amount)
    {
        MechBodyPart part = GetBodyPart(partType);
        if (part != null && !part.isDestroyed)
        {
            part.Repair(amount);
            TriggerDialogue("수리 완료", "고마워, 훨씬 나아졌어!");
        }
    }
    
    public MechBodyPart GetBodyPart(BodyPartType type)
    {
        return bodyParts.Find(part => part.partType == type);
    }
    
    public bool CanUseSkill(string skillName)
    {
        if (skillCooldowns.ContainsKey(skillName))
        {
            return skillCooldowns[skillName] <= 0;
        }
        return true;
    }
    
    public void UseSkill(string skillName, float cooldownTime)
    {
        skillCooldowns[skillName] = cooldownTime;
    }
    
    public void UpdateCooldowns(float deltaTime)
    {
        List<string> keys = new List<string>(skillCooldowns.Keys);
        foreach (string key in keys)
        {
            skillCooldowns[key] = Mathf.Max(0, skillCooldowns[key] - deltaTime);
        }
    }
    
    public void IncreaseTrust(MechType targetMech, int amount)
    {
        if (trustLevels.ContainsKey(targetMech))
        {
            trustLevels[targetMech] += amount;
            CheckTrustMilestones(targetMech);
        }
    }
    
    private void CheckTrustMilestones(MechType targetMech)
    {
        int trust = trustLevels[targetMech];
        if (trust >= 50 && trust < 100)
        {
            TriggerDialogue("신뢰도 상승", "우리 팀워크가 좋아지고 있어!");
        }
        else if (trust >= 100)
        {
            TriggerDialogue("최고 신뢰도", "이제 진짜 팀이 된 것 같아!");
        }
    }
    
    private void TriggerDamageDialogue(MechBodyPart part)
    {
        string dialogue = "";
        switch (part.GetDamageLevel())
        {
            case DamageLevel.Minor:
                dialogue = "괜찮아, 긁힌 정도야.";
                break;
            case DamageLevel.Moderate:
                dialogue = "이런... 움직임이 둔해졌어.";
                break;
            case DamageLevel.Severe:
                dialogue = "위험해! 코어가 노출됐어!";
                break;
            case DamageLevel.Critical:
                dialogue = "더 이상... 버틸 수가...";
                break;
        }
        
        if (!string.IsNullOrEmpty(dialogue))
        {
            TriggerDialogue("부상", dialogue);
        }
    }
    
    public void TriggerDialogue(string situation, string dialogue)
    {
        OnDialogueTriggered?.Invoke(this, $"[{situation}] {dialogue}");
    }
    
    public void StartTurn()
    {
        stats.currentAP = stats.maxAP;
        isGuarding = false;
        UpdateCooldowns(0);
    }
    
    public void EndTurn()
    {
        // 턴 종료 시 처리
    }
    
    public bool CanAct()
    {
        return isAlive && stats.currentAP > 0;
    }
    
    public void ConsumeAP(int amount)
    {
        stats.currentAP = Mathf.Max(0, stats.currentAP - amount);
    }
}


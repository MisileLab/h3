using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class CooperationSystem : MonoBehaviour
{
    [Header("협력 시스템")]
    public static CooperationSystem Instance;
    
    [Header("협력 스킬")]
    public List<CooperationSkill> availableSkills = new List<CooperationSkill>();
    
    // 이벤트
    public static event System.Action<MechCharacter, MechCharacter, string> OnCooperationUsed;
    public static event System.Action<MechCharacter, MechCharacter, int> OnTrustIncreased;
    
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    private void Start()
    {
        InitializeCooperationSkills();
    }
    
    private void InitializeCooperationSkills()
    {
        availableSkills.Clear();
        
        // 기본 협력 스킬들
        availableSkills.Add(new CooperationSkill(
            "가드",
            "동료를 보호합니다",
            1,
            new List<MechType> { MechType.Rex },
            CooperationType.Guard
        ));
        
        availableSkills.Add(new CooperationSkill(
            "응급처치",
            "동료의 부위를 수리합니다",
            2,
            new List<MechType> { MechType.Luna },
            CooperationType.EmergencyRepair
        ));
        
        availableSkills.Add(new CooperationSkill(
            "전술이동",
            "위치를 교환합니다",
            1,
            new List<MechType> { MechType.Zero },
            CooperationType.TacticalSwap
        ));
        
        availableSkills.Add(new CooperationSkill(
            "연계공격",
            "함께 공격합니다",
            2,
            new List<MechType> { MechType.Nova },
            CooperationType.LinkAttack
        ));
        
        availableSkills.Add(new CooperationSkill(
            "지원 사격",
            "동료의 공격을 지원합니다",
            1,
            new List<MechType> { MechType.Nova, MechType.Zero },
            CooperationType.SupportFire
        ));
        
        availableSkills.Add(new CooperationSkill(
            "합동 방어",
            "함께 방어합니다",
            2,
            new List<MechType> { MechType.Rex, MechType.Luna },
            CooperationType.CombinedDefense
        ));
    }
    
    public bool CanUseCooperation(MechCharacter user, MechCharacter target, string skillName)
    {
        CooperationSkill skill = availableSkills.Find(s => s.skillName == skillName);
        if (skill == null) return false;
        
        // AP 확인
        if (user.stats.currentAP < skill.apCost) return false;
        
        // 거리 확인
        float distance = Vector3.Distance(user.transform.position, target.transform.position);
        if (distance > skill.range) return false;
        
        // 신뢰도 확인
        int trustLevel = user.trustLevels.ContainsKey(target.mechType) ? user.trustLevels[target.mechType] : 0;
        if (trustLevel < skill.requiredTrust) return false;
        
        return true;
    }
    
    public void UseCooperation(MechCharacter user, MechCharacter target, string skillName)
    {
        CooperationSkill skill = availableSkills.Find(s => s.skillName == skillName);
        if (skill == null) return;
        
        if (!CanUseCooperation(user, target, skillName)) return;
        
        // AP 소모
        user.ConsumeAP(skill.apCost);
        
        // 협력 스킬 실행
        switch (skill.cooperationType)
        {
            case CooperationType.Guard:
                ExecuteGuard(user, target);
                break;
            case CooperationType.EmergencyRepair:
                ExecuteEmergencyRepair(user, target);
                break;
            case CooperationType.TacticalSwap:
                ExecuteTacticalSwap(user, target);
                break;
            case CooperationType.LinkAttack:
                ExecuteLinkAttack(user, target);
                break;
            case CooperationType.SupportFire:
                ExecuteSupportFire(user, target);
                break;
            case CooperationType.CombinedDefense:
                ExecuteCombinedDefense(user, target);
                break;
        }
        
        // 신뢰도 증가
        int trustGain = CalculateTrustGain(skill, user, target);
        user.IncreaseTrust(target.mechType, trustGain);
        
        OnCooperationUsed?.Invoke(user, target, skillName);
        OnTrustIncreased?.Invoke(user, target, trustGain);
    }
    
    private void ExecuteGuard(MechCharacter user, MechCharacter target)
    {
        // 가드: 동료를 보호
        target.isGuarding = true;
        
        // 렉스의 특수 가드 능력
        if (user.mechType == MechType.Rex)
        {
            RexMech rex = user as RexMech;
            if (rex != null)
            {
                rex.GuardAlly(target);
            }
        }
        
        Debug.Log($"{user.mechName}이 {target.mechName}을 보호합니다!");
        user.TriggerDialogue("가드", $"{target.mechName}! 내가 막을게!");
    }
    
    private void ExecuteEmergencyRepair(MechCharacter user, MechCharacter target)
    {
        // 응급처치: 가장 손상된 부위 수리
        MechBodyPart mostDamaged = GetMostDamagedPart(target);
        if (mostDamaged != null)
        {
            float repairAmount = 40f;
            target.RepairBodyPart(mostDamaged.partType, repairAmount);
            
            Debug.Log($"{user.mechName}이 {target.mechName}의 {mostDamaged.partName}을 수리했습니다!");
            user.TriggerDialogue("응급처치", "괜찮아, 내가 고쳐줄게!");
        }
    }
    
    private void ExecuteTacticalSwap(MechCharacter user, MechCharacter target)
    {
        // 전술이동: 위치 교환
        Vector3 userPos = user.transform.position;
        Vector3 targetPos = target.transform.position;
        
        user.transform.position = targetPos;
        target.transform.position = userPos;
        
        Debug.Log($"{user.mechName}과 {target.mechName}이 위치를 교환했습니다!");
        user.TriggerDialogue("전술이동", "위치 바꿔!");
    }
    
    private void ExecuteLinkAttack(MechCharacter user, MechCharacter target)
    {
        // 연계공격: 가장 가까운 적을 함께 공격
        EnemyAI nearestEnemy = FindNearestEnemy(user);
        if (nearestEnemy != null)
        {
            float combinedDamage = (user.stats.attack + target.stats.attack) * 0.8f;
            nearestEnemy.TakeDamage(combinedDamage);
            
            Debug.Log($"{user.mechName}과 {target.mechName}이 연계공격을 사용했습니다!");
            user.TriggerDialogue("연계공격", $"{target.mechName}과 함께!");
        }
    }
    
    private void ExecuteSupportFire(MechCharacter user, MechCharacter target)
    {
        // 지원 사격: 동료의 공격을 지원
        target.stats.accuracy += 25;
        target.stats.attack += 10;
        
        Debug.Log($"{user.mechName}이 {target.mechName}을 지원합니다!");
        user.TriggerDialogue("지원 사격", $"{target.mechName}, 지원할게!");
        
        // 2턴 후 효과 제거
        StartCoroutine(RemoveSupportFireAfterTime(target, 2f));
    }
    
    private void ExecuteCombinedDefense(MechCharacter user, MechCharacter target)
    {
        // 합동 방어: 둘 다 방어력 증가
        user.stats.defense += 15;
        target.stats.defense += 15;
        
        Debug.Log($"{user.mechName}과 {target.mechName}이 합동 방어를 사용했습니다!");
        user.TriggerDialogue("합동 방어", "함께 버티자!");
        
        // 3턴 후 효과 제거
        StartCoroutine(RemoveCombinedDefenseAfterTime(user, target, 3f));
    }
    
    private MechBodyPart GetMostDamagedPart(MechCharacter mech)
    {
        MechBodyPart mostDamaged = null;
        float lowestHP = float.MaxValue;
        
        foreach (MechBodyPart part in mech.bodyParts)
        {
            if (part.currentHP < lowestHP && !part.isDestroyed)
            {
                lowestHP = part.currentHP;
                mostDamaged = part;
            }
        }
        
        return mostDamaged;
    }
    
    private EnemyAI FindNearestEnemy(MechCharacter mech)
    {
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        EnemyAI nearest = null;
        float shortestDistance = float.MaxValue;
        
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive)
            {
                float distance = Vector3.Distance(mech.transform.position, enemy.transform.position);
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    nearest = enemy;
                }
            }
        }
        
        return nearest;
    }
    
    private int CalculateTrustGain(CooperationSkill skill, MechCharacter user, MechCharacter target)
    {
        int baseTrust = 5;
        
        // 스킬 타입에 따른 보너스
        switch (skill.cooperationType)
        {
            case CooperationType.Guard:
                baseTrust += 3; // 보호는 더 많은 신뢰도 증가
                break;
            case CooperationType.EmergencyRepair:
                baseTrust += 4; // 수리는 생명을 구하는 행위
                break;
            case CooperationType.LinkAttack:
                baseTrust += 2; // 전투 협력
                break;
        }
        
        // 기계 타입 조합에 따른 보너스
        if (IsGoodCombination(user.mechType, target.mechType))
        {
            baseTrust += 2;
        }
        
        return baseTrust;
    }
    
    private bool IsGoodCombination(MechType userType, MechType targetType)
    {
        // 좋은 조합들
        if (userType == MechType.Rex && targetType == MechType.Luna) return true; // 탱커-힐러
        if (userType == MechType.Luna && targetType == MechType.Rex) return true;
        if (userType == MechType.Zero && targetType == MechType.Nova) return true; // 스카우트-딜러
        if (userType == MechType.Nova && targetType == MechType.Zero) return true;
        
        return false;
    }
    
    private IEnumerator RemoveSupportFireAfterTime(MechCharacter target, float time)
    {
        yield return new WaitForSeconds(time);
        target.stats.accuracy -= 25;
        target.stats.attack -= 10;
        Debug.Log($"{target.mechName}의 지원 사격 효과가 사라졌습니다.");
    }
    
    private IEnumerator RemoveCombinedDefenseAfterTime(MechCharacter user, MechCharacter target, float time)
    {
        yield return new WaitForSeconds(time);
        user.stats.defense -= 15;
        target.stats.defense -= 15;
        Debug.Log("합동 방어 효과가 사라졌습니다.");
    }
    
    public List<CooperationSkill> GetAvailableSkillsForMech(MechCharacter mech)
    {
        List<CooperationSkill> available = new List<CooperationSkill>();
        
        foreach (CooperationSkill skill in availableSkills)
        {
            if (skill.requiredMechs.Contains(mech.mechType))
            {
                available.Add(skill);
            }
        }
        
        return available;
    }
    
    public bool CanPerformCooperation(MechCharacter user, MechCharacter target)
    {
        List<CooperationSkill> userSkills = GetAvailableSkillsForMech(user);
        
        foreach (CooperationSkill skill in userSkills)
        {
            if (CanUseCooperation(user, target, skill.skillName))
            {
                return true;
            }
        }
        
        return false;
    }
}

[System.Serializable]
public class CooperationSkill
{
    public string skillName;
    public string description;
    public int apCost;
    public List<MechType> requiredMechs;
    public CooperationType cooperationType;
    public float range = 3f;
    public int requiredTrust = 0;
    
    public CooperationSkill(string name, string desc, int cost, List<MechType> mechs, CooperationType type)
    {
        skillName = name;
        description = desc;
        apCost = cost;
        requiredMechs = mechs;
        cooperationType = type;
    }
}

public enum CooperationType
{
    Guard,              // 가드
    EmergencyRepair,    // 응급처치
    TacticalSwap,       // 전술이동
    LinkAttack,         // 연계공격
    SupportFire,        // 지원 사격
    CombinedDefense     // 합동 방어
}

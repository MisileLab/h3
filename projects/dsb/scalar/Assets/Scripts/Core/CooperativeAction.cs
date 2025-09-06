using UnityEngine;
using System;
using System.Collections.Generic;

/// <summary>
/// 협력 시스템의 기본 클래스
/// 두 명 이상의 기계가 상호작용하여 발동하는 특별한 행동들을 관리합니다.
/// </summary>
public abstract class CooperativeAction
{
    [Header("협력 행동 정보")]
    public string actionName;
    public string description;
    public int apCost = 2;                          // 협력 행동은 기본적으로 2 AP 소모
    public int maxRange = 2;                        // 최대 사용 거리
    public float cooldown = 0f;                     // 쿨다운 (턴 단위)
    public List<MechCharacter> requiredMechs;       // 필요한 기계들
    
    // 이벤트
    public static event Action<CooperativeAction, List<MechCharacter>> OnCooperativeActionPerformed;
    
    public CooperativeAction()
    {
        requiredMechs = new List<MechCharacter>();
    }
    
    /// <summary>
    /// 협력 행동을 수행할 수 있는지 확인합니다
    /// </summary>
    /// <param name="actors">행동을 수행하려는 기계들</param>
    /// <returns>수행 가능하면 true</returns>
    public virtual bool CanPerform(List<MechCharacter> actors)
    {
        // 기본 조건 확인
        if (actors == null || actors.Count < 2) return false;
        
        // 모든 기계가 살아있고 AP가 충분한지 확인
        foreach (var mech in actors)
        {
            if (!mech.isAlive) return false;
            if (!mech.actionPoints.CanUseAP(apCost)) return false;
        }
        
        // 거리 확인
        if (!IsWithinRange(actors)) return false;
        
        // 쿨다운 확인
        if (cooldown > 0f) return false;
        
        return true;
    }
    
    /// <summary>
    /// 협력 행동을 실행합니다
    /// </summary>
    /// <param name="actors">행동을 수행하는 기계들</param>
    /// <param name="target">대상 (옵션)</param>
    /// <returns>성공하면 true</returns>
    public virtual bool Perform(List<MechCharacter> actors, object target = null)
    {
        if (!CanPerform(actors)) return false;
        
        // AP 소모
        foreach (var mech in actors)
        {
            mech.actionPoints.ConsumeAP(apCost);
        }
        
        // 실제 행동 수행
        bool success = ExecuteAction(actors, target);
        
        if (success)
        {
            // 쿨다운 설정
            SetCooldown();
            
            // 이벤트 발생
            OnCooperativeActionPerformed?.Invoke(this, actors);
            
            // 신뢰도 상승
            foreach (var mech in actors)
            {
                mech.IncreaseTrustWithAllies(actors, 5);
            }
            
            Debug.Log($"<color=green>협력 행동 성공: {actionName}</color>");
        }
        
        return success;
    }
    
    /// <summary>
    /// 실제 협력 행동을 실행하는 추상 메서드
    /// </summary>
    /// <param name="actors">행동자들</param>
    /// <param name="target">대상</param>
    /// <returns>성공 여부</returns>
    protected abstract bool ExecuteAction(List<MechCharacter> actors, object target);
    
    /// <summary>
    /// 행동자들이 유효한 범위 내에 있는지 확인
    /// </summary>
    /// <param name="actors">행동자들</param>
    /// <returns>범위 내에 있으면 true</returns>
    protected virtual bool IsWithinRange(List<MechCharacter> actors)
    {
        if (actors.Count < 2) return false;
        
        // 모든 기계가 서로 maxRange 내에 있는지 확인
        for (int i = 0; i < actors.Count; i++)
        {
            for (int j = i + 1; j < actors.Count; j++)
            {
                float distance = Vector3.Distance(actors[i].transform.position, actors[j].transform.position);
                if (distance > maxRange)
                {
                    Debug.LogWarning($"{actors[i].mechName}과 {actors[j].mechName}의 거리가 너무 멉니다. ({distance:F1} > {maxRange})");
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// 쿨다운을 설정합니다
    /// </summary>
    protected virtual void SetCooldown()
    {
        // 기본 구현: 쿨다운 없음
        // 하위 클래스에서 필요에 따라 오버라이드
    }
    
    /// <summary>
    /// 쿨다운을 업데이트합니다 (턴 종료 시 호출)
    /// </summary>
    /// <param name="turns">경과한 턴 수</param>
    public virtual void UpdateCooldown(int turns = 1)
    {
        if (cooldown > 0)
        {
            cooldown -= turns;
            if (cooldown <= 0)
            {
                cooldown = 0;
                Debug.Log($"{actionName} 쿨다운이 완료되었습니다.");
            }
        }
    }
}

/// <summary>
/// 보호하기: 동료에게 가해지는 공격을 대신 맞습니다
/// 렉스의 핵심 능력이지만, 다른 기계도 사용 가능합니다.
/// </summary>
public class GuardAction : CooperativeAction
{
    public GuardAction()
    {
        actionName = "보호하기";
        description = "동료에게 가해지는 다음 공격을 대신 받습니다.";
        apCost = 2;
        maxRange = 1; // 인접해야 함
    }
    
    protected override bool ExecuteAction(List<MechCharacter> actors, object target)
    {
        if (actors.Count != 2) return false;
        
        MechCharacter protector = actors[0];    // 보호하는 자
        MechCharacter protected_mech = actors[1]; // 보호받는 자
        
        // 보호 상태 설정
        protected_mech.SetProtector(protector, 1); // 1턴 동안 보호
        
        // 보호자의 대사
        protector.SayDialogue($"{protected_mech.mechName}! 내 뒤로 숨어!", DialogueType.Cooperative);
        protected_mech.SayDialogue($"{protector.mechName}, 고마워!", DialogueType.Grateful);
        
        return true;
    }
}

/// <summary>
/// 연계 공격: 특정 조건을 만족한 두 기계가 하나의 적을 동시에 공격
/// </summary>
public class LinkAttackAction : CooperativeAction
{
    public float damageMultiplier = 1.5f; // 피해 배율
    
    public LinkAttackAction()
    {
        actionName = "연계 공격";
        description = "두 기계가 협력하여 강력한 연계 공격을 가합니다.";
        apCost = 1; // 각자 1 AP씩 소모 (총 2 AP)
        maxRange = 3;
        cooldown = 2f; // 2턴 쿨다운
    }
    
    protected override bool ExecuteAction(List<MechCharacter> actors, object target)
    {
        if (actors.Count != 2) return false;
        if (target == null || !(target is EnemyAI enemy)) return false;
        
        MechCharacter attacker1 = actors[0];
        MechCharacter attacker2 = actors[1];
        
        // 각 기계의 기본 공격력을 합산하고 배율 적용
        int totalDamage = Mathf.RoundToInt((attacker1.stats.attack + attacker2.stats.attack) * damageMultiplier);
        
        // 적에게 피해 가하기
        enemy.TakeDamage(totalDamage);
        
        // 연계 공격 대사
        attacker1.SayDialogue($"{attacker2.mechName}, 지금이야!", DialogueType.Cooperative);
        attacker2.SayDialogue("좋은 타이밍이야!", DialogueType.Cooperative);
        
        Debug.Log($"<color=yellow>연계 공격! {attacker1.mechName} + {attacker2.mechName} → {enemy.enemyName} ({totalDamage} 피해)</color>");
        
        return true;
    }
    
    protected override void SetCooldown()
    {
        cooldown = 2f; // 2턴 쿨다운
    }
}

/// <summary>
/// 응급 처치: 전투 중 동료의 부위 손상을 임시로 복구
/// </summary>
public class FirstAidAction : CooperativeAction
{
    public int healAmount = 30; // 치료량
    
    public FirstAidAction()
    {
        actionName = "응급 처치";
        description = "동료의 손상을 응급 처치하여 일부 기능을 복구합니다.";
        apCost = 2;
        maxRange = 1; // 인접해야 함
    }
    
    protected override bool ExecuteAction(List<MechCharacter> actors, object target)
    {
        if (actors.Count != 2) return false;
        
        MechCharacter medic = actors[0];        // 치료하는 자
        MechCharacter patient = actors[1];      // 치료받는 자
        
        // 가장 손상이 심한 부위를 찾아서 치료
        BodyPart mostDamagedPart = patient.GetMostDamagedPart();
        if (mostDamagedPart != null)
        {
            int actualHeal = mostDamagedPart.Heal(healAmount, true); // 파괴된 부위도 임시 수리 가능
            
            medic.SayDialogue($"{patient.mechName}, 괜찮아! 내가 고쳐줄게!", DialogueType.Cooperative);
            patient.SayDialogue($"고마워, {medic.mechName}. 훨씬 나아졌어.", DialogueType.Grateful);
            
            Debug.Log($"<color=green>응급 처치 완료: {patient.mechName}의 {mostDamagedPart.partName} {actualHeal} 치료</color>");
            return true;
        }
        
        return false;
    }
}

/// <summary>
/// 전술 이동: 인접한 아군과 위치를 즉시 교환
/// </summary>
public class TacticalSwapAction : CooperativeAction
{
    public TacticalSwapAction()
    {
        actionName = "전술 이동";
        description = "동료와 위치를 즉시 교환하여 전략적 위치를 선점합니다.";
        apCost = 1; // 빠른 이동이므로 저렴
        maxRange = 1; // 인접해야만 교환 가능
    }
    
    protected override bool ExecuteAction(List<MechCharacter> actors, object target)
    {
        if (actors.Count != 2) return false;
        
        MechCharacter mech1 = actors[0];
        MechCharacter mech2 = actors[1];
        
        // 위치 교환
        Vector3 temp = mech1.transform.position;
        mech1.transform.position = mech2.transform.position;
        mech2.transform.position = temp;
        
        // 위치 교환 대사
        mech1.SayDialogue($"{mech2.mechName}, 위치 바꾸자!", DialogueType.Cooperative);
        mech2.SayDialogue("알았어, 지금!", DialogueType.Cooperative);
        
        Debug.Log($"<color=cyan>전술 이동: {mech1.mechName} ↔ {mech2.mechName}</color>");
        
        return true;
    }
}

/// <summary>
/// 협력 행동 매니저
/// 사용 가능한 협력 행동들을 관리하고 실행을 담당합니다.
/// </summary>
public class CooperativeActionManager : MonoBehaviour
{
    [Header("사용 가능한 협력 행동")]
    public List<CooperativeAction> availableActions;
    
    private void Start()
    {
        InitializeActions();
    }
    
    private void InitializeActions()
    {
        availableActions = new List<CooperativeAction>
        {
            new GuardAction(),
            new LinkAttackAction(),
            new FirstAidAction(),
            new TacticalSwapAction()
        };
        
        Debug.Log($"협력 행동 시스템 초기화 완료: {availableActions.Count}개 행동 등록");
    }
    
    /// <summary>
    /// 지정된 기계들이 수행할 수 있는 협력 행동 목록을 반환
    /// </summary>
    /// <param name="actors">행동자들</param>
    /// <returns>수행 가능한 협력 행동들</returns>
    public List<CooperativeAction> GetAvailableActions(List<MechCharacter> actors)
    {
        List<CooperativeAction> availableForActors = new List<CooperativeAction>();
        
        foreach (var action in availableActions)
        {
            if (action.CanPerform(actors))
            {
                availableForActors.Add(action);
            }
        }
        
        return availableForActors;
    }
    
    /// <summary>
    /// 모든 협력 행동의 쿨다운을 업데이트 (턴 종료 시 호출)
    /// </summary>
    public void UpdateAllCooldowns()
    {
        foreach (var action in availableActions)
        {
            action.UpdateCooldown();
        }
    }
}

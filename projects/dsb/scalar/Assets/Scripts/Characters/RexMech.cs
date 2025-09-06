using UnityEngine;
using System.Collections.Generic;
using System.Collections;

/// <summary>
/// 렉스 (Rex) - 프론트라인 가디언 (탱커)
/// 핵심 능력: 가디언 실드 - 자신 또는 아군에게 가해지는 피해를 흡수하는 방패 생성
/// 전투 철학: "뒤로 물러나! 내가 막을게!"
/// </summary>
public class RexMech : MechCharacter
{
    [Header("렉스 전용 설정")]
    public int shieldMaxHP = 50;                    // 방패 최대 HP
    public int currentShieldHP = 0;                 // 현재 방패 HP
    public float shieldDuration = 3f;               // 방패 지속 시간 (턴)
    public float shieldCooldown = 0f;               // 방패 쿨다운
    public float shieldMaxCooldown = 2f;            // 방패 최대 쿨다운
    public MechCharacter shieldTarget;              // 방패로 보호 중인 대상
    public int shieldTurnsLeft = 0;                 // 방패 지속 턴
    
    [Header("렉스 고유 능력")]
    public float tauntChance = 0.8f;                // 도발 확률
    public int damageReduction = 15;                // 받는 피해 감소 (%)
    public float counterAttackChance = 0.3f;        // 반격 확률
    
    protected override void Start()
    {
        // 기본 정보 설정
        mechName = "렉스";
        mechType = MechType.Rex;
        
        // 렉스의 스탯 (탱커 특화)
        stats = new MechStats
        {
            maxHP = 120,        // 높은 HP
            currentHP = 120,
            attack = 25,        // 중간 공격력
            defense = 30,       // 높은 방어력
            speed = 6,          // 낮은 속도
            accuracy = 75,      // 보통 명중률
            evasion = 10        // 낮은 회피율
        };
        
        base.Start();
    }
    
    protected override void InitializeDialogues()
    {
        dialogueLines = new List<DialogueLine>
        {
            // 턴 시작
            new DialogueLine { type = DialogueType.TurnStart, text = "내가 앞장설게!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "모두 내 뒤에 숨어!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "지킬 것이 있으니까 강해질 수 있어.", weight = 0.5f },
            
            // 피해 받을 때
            new DialogueLine { type = DialogueType.Damage, text = "이 정도론 안 쓰러져!", weight = 1f },
            new DialogueLine { type = DialogueType.Damage, text = "아직 괜찮아!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "으윽... 하지만 아직 설 수 있어!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "동료들을 지켜야 해!", weight = 1f },
            
            // 치료 받을 때
            new DialogueLine { type = DialogueType.Healed, text = "고마워! 이제 더 오래 버틸 수 있어.", weight = 1f },
            new DialogueLine { type = DialogueType.Healed, text = "다시 전열에 설 수 있겠어.", weight = 1f },
            
            // 협력 행동
            new DialogueLine { type = DialogueType.Cooperative, text = "뒤로 물러나! 내가 막을게!", weight = 1f },
            new DialogueLine { type = DialogueType.Protective, text = "내가 지킬게, 걱정하지 마!", weight = 1f },
            new DialogueLine { type = DialogueType.Protective, text = "누구든 먼저 나를 넘어가라!", weight = 1f },
            
            // 부위 파괴
            new DialogueLine { type = DialogueType.PartDestroyed, text = "아직... 아직 끝나지 않았어!", weight = 1f },
            
            // 전투 불능
            new DialogueLine { type = DialogueType.Incapacitated, text = "미안해... 더 이상 지켜줄 수 없어...", weight = 1f },
            new DialogueLine { type = DialogueType.Incapacitated, text = "모두... 무사히... 집에 가...", weight = 1f },
            
            // 승리
            new DialogueLine { type = DialogueType.Victory, text = "모두 무사해서 다행이야!", weight = 1f },
            new DialogueLine { type = DialogueType.Victory, text = "함께라면 어떤 것이든 이겨낼 수 있어!", weight = 1f }
        };
    }
    
    public override void StartTurn()
    {
        base.StartTurn();
        
        // 방패 지속 시간 감소
        if (shieldTurnsLeft > 0)
        {
            shieldTurnsLeft--;
            if (shieldTurnsLeft <= 0)
            {
                DeactivateShield();
            }
        }
    }
    
    public override void UpdateCooldowns(float deltaTime)
    {
        base.UpdateCooldowns(deltaTime);
        
        // 방패 쿨다운 업데이트
        if (shieldCooldown > 0)
        {
            shieldCooldown -= deltaTime;
            if (shieldCooldown <= 0)
            {
                shieldCooldown = 0;
                Debug.Log($"{mechName}의 가디언 실드 쿨다운이 완료되었습니다.");
            }
        }
    }
    
    /// <summary>
    /// 가디언 실드 능력 - 자신 또는 아군을 보호하는 방패를 생성
    /// </summary>
    /// <param name="target">보호할 대상 (null이면 자신)</param>
    /// <returns>성공하면 true</returns>
    public bool UseGuardianShield(MechCharacter target = null)
    {
        // AP 및 쿨다운 확인
        if (!actionPoints.CanUseAP(2)) 
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 2)");
            return false;
        }
        
        if (shieldCooldown > 0)
        {
            Debug.LogWarning($"가디언 실드가 쿨다운 중입니다. ({shieldCooldown:F1}초 남음)");
            return false;
        }
        
        // 대상 설정 (null이면 자신)
        if (target == null) target = this;
        
        // AP 소모
        actionPoints.ConsumeAP(2);
        
        // 방패 활성화
        ActivateShield(target);
        
        // 쿨다운 설정
        shieldCooldown = shieldMaxCooldown;
        
        // 대사
        if (target == this)
        {
            SayDialogue("방어 태세 완료!", DialogueType.Cooperative);
        }
        else
        {
            SayDialogue($"{target.mechName}, 내가 지킬게!", DialogueType.Protective);
        }
        
        Debug.Log($"<color=blue>{mechName}이 {target.mechName}에게 가디언 실드를 사용했습니다!</color>");
        return true;
    }
    
    /// <summary>
    /// 도발 능력 - 적들이 렉스를 우선적으로 공격하도록 유도
    /// </summary>
    /// <param name="enemies">도발할 적들</param>
    /// <returns>성공하면 true</returns>
    public bool UseTaunt(List<EnemyAI> enemies)
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        actionPoints.ConsumeAP(1);
        
        int taunted = 0;
        foreach (var enemy in enemies)
        {
            if (enemy.isAlive && UnityEngine.Random.Range(0f, 1f) < tauntChance)
            {
                enemy.SetTauntedTarget(this, 2f); // 2턴 동안 도발
                taunted++;
            }
        }
        
        if (taunted > 0)
        {
            SayDialogue("이리 와! 상대는 나다!", DialogueType.Cooperative);
            Debug.Log($"<color=yellow>{mechName}이 {taunted}명의 적을 도발했습니다!</color>");
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// 방패 활성화
    /// </summary>
    /// <param name="target">보호할 대상</param>
    private void ActivateShield(MechCharacter target)
    {
        shieldTarget = target;
        currentShieldHP = shieldMaxHP;
        shieldTurnsLeft = Mathf.RoundToInt(shieldDuration);
        
        Debug.Log($"{mechName}의 방패가 활성화되었습니다. (대상: {target.mechName}, HP: {currentShieldHP}, 지속: {shieldTurnsLeft}턴)");
    }
    
    /// <summary>
    /// 방패 비활성화
    /// </summary>
    private void DeactivateShield()
    {
        if (shieldTarget != null)
        {
            Debug.Log($"{shieldTarget.mechName}의 방패가 해제되었습니다.");
            shieldTarget = null;
        }
        currentShieldHP = 0;
        shieldTurnsLeft = 0;
    }
    
    /// <summary>
    /// 피해를 받습니다 (방패 시스템 적용)
    /// </summary>
    public override int TakeDamage(int damage, BodyPartType? targetPart = null)
    {
        // 방패가 활성화된 경우 방패가 먼저 피해를 받음
        if (currentShieldHP > 0)
        {
            int shieldDamage = Mathf.Min(damage, currentShieldHP);
            currentShieldHP -= shieldDamage;
            damage -= shieldDamage;
            
            Debug.Log($"<color=cyan>{mechName}의 방패가 {shieldDamage} 피해를 흡수했습니다. (방패 HP: {currentShieldHP})</color>");
            
            if (currentShieldHP <= 0)
            {
                SayDialogue("방패가 부서졌어! 하지만 아직 버틸 수 있어!", DialogueType.Damage);
                DeactivateShield();
            }
            
            // 방패가 모든 피해를 흡수한 경우
            if (damage <= 0) return shieldDamage;
        }
        
        // 렉스의 피해 감소 적용
        damage = Mathf.RoundToInt(damage * (1f - damageReduction / 100f));
        
        // 기본 피해 처리
        int actualDamage = base.TakeDamage(damage, targetPart);
        
        // 반격 확률 체크
        if (actualDamage > 0 && UnityEngine.Random.Range(0f, 1f) < counterAttackChance)
        {
            StartCoroutine(CounterAttack());
        }
        
        return actualDamage;
    }
    
    /// <summary>
    /// 반격 코루틴
    /// </summary>
    private IEnumerator CounterAttack()
    {
        yield return new WaitForSeconds(0.5f);
        
        // 가장 가까운 적을 찾아서 반격
        EnemyAI nearestEnemy = FindNearestEnemy();
        if (nearestEnemy != null)
        {
            SayDialogue("반격이다!", DialogueType.Cooperative);
            int counterDamage = Mathf.RoundToInt(stats.attack * 0.7f); // 70% 위력으로 반격
            nearestEnemy.TakeDamage(counterDamage);
            Debug.Log($"<color=orange>{mechName}이 {nearestEnemy.enemyName}에게 반격했습니다! ({counterDamage} 피해)</color>");
        }
    }
    
    /// <summary>
    /// 가장 가까운 적을 찾습니다
    /// </summary>
    /// <returns>가장 가까운 적</returns>
    private EnemyAI FindNearestEnemy()
    {
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        EnemyAI nearest = null;
        float shortestDistance = float.MaxValue;
        
        foreach (var enemy in enemies)
        {
            if (enemy.isAlive)
            {
                float distance = Vector3.Distance(transform.position, enemy.transform.position);
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    nearest = enemy;
                }
            }
        }
        
        return nearest;
    }
    
    /// <summary>
    /// 현재 방패로 보호 중인 대상이 피해를 받을 때 호출
    /// </summary>
    /// <param name="damage">피해량</param>
    /// <param name="target">피해를 받는 대상</param>
    /// <returns>방패가 흡수한 피해량</returns>
    public int AbsorbDamageForTarget(int damage, MechCharacter target)
    {
        if (shieldTarget != target || currentShieldHP <= 0) return 0;
        
        int absorbed = Mathf.Min(damage, currentShieldHP);
        currentShieldHP -= absorbed;
        
        Debug.Log($"<color=cyan>{mechName}의 방패가 {target.mechName}을 위해 {absorbed} 피해를 흡수했습니다!</color>");
        
        if (currentShieldHP <= 0)
        {
            SayDialogue($"{target.mechName}의 방패가 부서졌어! 조심해!", DialogueType.Worried);
            DeactivateShield();
        }
        
        return absorbed;
    }
    
    /// <summary>
    /// 렉스의 리더십 능력 - 주변 아군의 사기를 높임
    /// </summary>
    /// <returns>영향받은 아군 수</returns>
    public int UseLeadership()
    {
        if (!actionPoints.CanUseAP(1)) return 0;
        
        actionPoints.ConsumeAP(1);
        
        var allies = FindObjectsOfType<MechCharacter>();
        int affected = 0;
        
        foreach (var ally in allies)
        {
            if (ally != this && ally.isAlive)
            {
                float distance = Vector3.Distance(transform.position, ally.transform.position);
                if (distance <= 3f) // 3칸 내의 아군
                {
                    // 다음 턴에 추가 AP 부여 (임시 버프)
                    ally.actionPoints.RecoverAP(1);
                    affected++;
                }
            }
        }
        
        if (affected > 0)
        {
            SayDialogue("모두 힘내! 우리는 해낼 수 있어!", DialogueType.Cooperative);
            Debug.Log($"<color=green>{mechName}의 리더십으로 {affected}명의 아군이 사기를 얻었습니다!</color>");
        }
        
        return affected;
    }
    
    /// <summary>
    /// 상태 정보를 문자열로 반환
    /// </summary>
    /// <returns>상태 문자열</returns>
    public override string ToString()
    {
        string status = $"{mechName} (HP: {stats.currentHP}/{stats.maxHP}, AP: {actionPoints.currentAP}/{actionPoints.maxAP})";
        
        if (currentShieldHP > 0)
        {
            status += $"\n방패: {currentShieldHP}/{shieldMaxHP} ({shieldTurnsLeft}턴)";
        }
        
        if (shieldCooldown > 0)
        {
            status += $"\n가디언 실드 쿨다운: {shieldCooldown:F1}초";
        }
        
        return status;
    }
}

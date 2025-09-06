using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

/// <summary>
/// 루나 (Luna) - 테크니컬 서포터 (힐러/해커)
/// 핵심 능력: 나노 리페어 - 아군의 HP를 회복하고 파괴된 부위를 임시 수복
/// 전투 철학: "괜찮아, 내가 고쳐줄게!"
/// </summary>
public class LunaMech : MechCharacter
{
    [Header("루나 전용 설정")]
    public int repairPower = 35;                    // 수리 능력
    public float repairRange = 3f;                  // 수리 사거리
    public float repairCooldown = 0f;               // 수리 쿨다운
    public float repairMaxCooldown = 1f;            // 수리 최대 쿨다운
    
    [Header("해킹 능력")]
    public float hackRange = 4f;                    // 해킹 사거리
    public float hackSuccessRate = 0.75f;           // 해킹 성공률
    public float hackDuration = 3f;                 // 해킹 지속 시간 (턴)
    public float hackCooldown = 0f;                 // 해킹 쿨다운
    public float hackMaxCooldown = 3f;              // 해킹 최대 쿨다운
    
    [Header("분석 및 정보 수집")]
    public List<EnemyAI> analyzedEnemies;           // 분석된 적들
    public float scanRange = 5f;                    // 스캔 사거리
    public float shieldRegenRate = 5f;              // 방패 재생 속도 (턴당)
    
    protected override void Start()
    {
        // 기본 정보 설정
        mechName = "루나";
        mechType = MechType.Luna;
        
        // 루나의 스탯 (서포터 특화)
        stats = new MechStats
        {
            maxHP = 80,         // 낮은 HP (서포터)
            currentHP = 80,
            attack = 15,        // 낮은 공격력
            defense = 15,       // 낮은 방어력
            speed = 12,         // 높은 속도
            accuracy = 85,      // 높은 명중률 (정밀 작업)
            evasion = 25        // 높은 회피율
        };
        
        analyzedEnemies = new List<EnemyAI>();
        
        base.Start();
    }
    
    protected override void InitializeDialogues()
    {
        dialogueLines = new List<DialogueLine>
        {
            // 턴 시작
            new DialogueLine { type = DialogueType.TurnStart, text = "시스템 체크 완료!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "누구 도움이 필요해?", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "데이터 분석 중...", weight = 0.8f },
            
            // 피해 받을 때
            new DialogueLine { type = DialogueType.Damage, text = "앗, 시스템에 오류가!", weight = 1f },
            new DialogueLine { type = DialogueType.Damage, text = "이런... 회로가 손상됐어.", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "주요 시스템이 다운되고 있어!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "누군가 도움이... 필요해...", weight = 1f },
            
            // 치료 받을 때
            new DialogueLine { type = DialogueType.Healed, text = "시스템 복구 중... 고마워!", weight = 1f },
            new DialogueLine { type = DialogueType.Healed, text = "다시 정상 작동이 가능해졌어.", weight = 1f },
            
            // 협력 행동 (치료/수리 시)
            new DialogueLine { type = DialogueType.Cooperative, text = "괜찮아, 내가 고쳐줄게!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "잠깐만, 나노봇들이 수리 중이야.", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "시스템 복구 완료!", weight = 1f },
            
            // 해킹 성공 시
            new DialogueLine { type = DialogueType.Cooperative, text = "해킹 성공! 적 시스템을 장악했어!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "적의 보안이 뚫렸어!", weight = 1f },
            
            // 분석 완료 시
            new DialogueLine { type = DialogueType.Cooperative, text = "적의 약점을 발견했어!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "데이터 분석 완료! 전술 정보를 공유할게.", weight = 1f },
            
            // 부위 파괴
            new DialogueLine { type = DialogueType.PartDestroyed, text = "시스템 오류 발생!", weight = 1f },
            
            // 전투 불능
            new DialogueLine { type = DialogueType.Incapacitated, text = "시스템... 정지... 모두... 조심해...", weight = 1f },
            new DialogueLine { type = DialogueType.Incapacitated, text = "데이터... 백업... 완료...", weight = 1f },
            
            // 회복
            new DialogueLine { type = DialogueType.Revived, text = "시스템 재부팅... 완료!", weight = 1f },
            new DialogueLine { type = DialogueType.Revived, text = "백업에서 복구 성공!", weight = 1f },
            
            // 승리
            new DialogueLine { type = DialogueType.Victory, text = "모든 시스템 정상! 임무 완료!", weight = 1f },
            new DialogueLine { type = DialogueType.Victory, text = "데이터 수집 완료. 집에 갈 시간이야!", weight = 1f }
        };
    }
    
    public override void UpdateCooldowns(float deltaTime)
    {
        base.UpdateCooldowns(deltaTime);
        
        // 수리 쿨다운 업데이트
        if (repairCooldown > 0)
        {
            repairCooldown -= deltaTime;
            if (repairCooldown <= 0)
            {
                repairCooldown = 0;
                Debug.Log($"{mechName}의 나노 리페어 쿨다운이 완료되었습니다.");
            }
        }
        
        // 해킹 쿨다운 업데이트
        if (hackCooldown > 0)
        {
            hackCooldown -= deltaTime;
            if (hackCooldown <= 0)
            {
                hackCooldown = 0;
                Debug.Log($"{mechName}의 해킹 능력 쿨다운이 완료되었습니다.");
            }
        }
    }
    
    /// <summary>
    /// 나노 리페어 능력 - 아군의 HP를 회복하고 파괴된 부위를 임시 수복
    /// </summary>
    /// <param name="target">치료할 대상</param>
    /// <param name="targetPart">특정 부위 치료 (null이면 자동 선택)</param>
    /// <returns>성공하면 true</returns>
    public bool UseNanoRepair(MechCharacter target, BodyPartType? targetPart = null)
    {
        // AP 및 쿨다운 확인
        if (!actionPoints.CanUseAP(2))
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 2)");
            return false;
        }
        
        if (repairCooldown > 0)
        {
            Debug.LogWarning($"나노 리페어가 쿨다운 중입니다. ({repairCooldown:F1}초 남음)");
            return false;
        }
        
        // 거리 확인
        float distance = Vector3.Distance(transform.position, target.transform.position);
        if (distance > repairRange)
        {
            Debug.LogWarning($"치료 대상이 너무 멀리 있습니다. (거리: {distance:F1}, 최대: {repairRange})");
            return false;
        }
        
        // AP 소모
        actionPoints.ConsumeAP(2);
        
        // 치료 실행
        int healAmount = repairPower;
        
        // 대상 부위 결정
        BodyPart healTarget;
        if (targetPart.HasValue)
        {
            healTarget = target.bodyParts.FirstOrDefault(bp => bp.partType == targetPart.Value);
        }
        else
        {
            healTarget = target.GetMostDamagedPart();
        }
        
        if (healTarget != null)
        {
            int actualHeal = healTarget.Heal(healAmount, true); // 파괴된 부위도 임시 수리 가능
            
            if (actualHeal > 0)
            {
                // 치료 대사
                SayDialogue($"{target.mechName}, 괜찮아! 내가 고쳐줄게!", DialogueType.Cooperative);
                
                if (target != this)
                {
                    target.SayDialogue($"고마워, {mechName}! 훨씬 나아졌어.", DialogueType.Grateful);
                }
                
                // 쿨다운 설정
                repairCooldown = repairMaxCooldown;
                
                Debug.Log($"<color=green>{mechName}이 {target.mechName}의 {healTarget.partName}을 {actualHeal} 치료했습니다!</color>");
                return true;
            }
        }
        
        Debug.LogWarning("치료할 대상을 찾을 수 없습니다.");
        return false;
    }
    
    /// <summary>
    /// 해킹 능력 - 적을 일시적으로 무력화하거나 아군으로 만들기
    /// </summary>
    /// <param name="target">해킹할 적</param>
    /// <returns>성공하면 true</returns>
    public bool UseHacking(EnemyAI target)
    {
        // AP 및 쿨다운 확인
        if (!actionPoints.CanUseAP(2))
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 2)");
            return false;
        }
        
        if (hackCooldown > 0)
        {
            Debug.LogWarning($"해킹 능력이 쿨다운 중입니다. ({hackCooldown:F1}초 남음)");
            return false;
        }
        
        // 거리 확인
        float distance = Vector3.Distance(transform.position, target.transform.position);
        if (distance > hackRange)
        {
            Debug.LogWarning($"해킹 대상이 너무 멀리 있습니다. (거리: {distance:F1}, 최대: {hackRange})");
            return false;
        }
        
        // 해킹 성공률 체크
        bool success = UnityEngine.Random.Range(0f, 1f) < hackSuccessRate;
        
        // AP 소모
        actionPoints.ConsumeAP(2);
        
        if (success)
        {
            // 해킹 성공
            target.GetHacked(hackDuration);
            
            SayDialogue("해킹 성공! 적 시스템을 장악했어!", DialogueType.Cooperative);
            
            // 쿨다운 설정
            hackCooldown = hackMaxCooldown;
            
            Debug.Log($"<color=purple>{mechName}이 {target.enemyName}을 해킹했습니다! ({hackDuration}턴 지속)</color>");
            return true;
        }
        else
        {
            // 해킹 실패
            SayDialogue("해킹 실패... 보안이 더 강화됐어.", DialogueType.Damage);
            
            // 실패해도 절반 쿨다운
            hackCooldown = hackMaxCooldown * 0.5f;
            
            Debug.Log($"<color=red>{mechName}의 해킹이 실패했습니다.</color>");
            return false;
        }
    }
    
    /// <summary>
    /// 적 분석 능력 - 적의 약점과 정보를 파악
    /// </summary>
    /// <param name="target">분석할 적</param>
    /// <returns>성공하면 true</returns>
    public bool UseEnemyAnalysis(EnemyAI target)
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        // 거리 확인
        float distance = Vector3.Distance(transform.position, target.transform.position);
        if (distance > scanRange)
        {
            Debug.LogWarning($"스캔 대상이 너무 멀리 있습니다. (거리: {distance:F1}, 최대: {scanRange})");
            return false;
        }
        
        actionPoints.ConsumeAP(1);
        
        // 적 정보 노출
        target.RevealInfo();
        
        // 분석된 적 목록에 추가
        if (!analyzedEnemies.Contains(target))
        {
            analyzedEnemies.Add(target);
        }
        
        SayDialogue("적의 약점을 발견했어!", DialogueType.Cooperative);
        
        Debug.Log($"<color=yellow>{mechName}이 {target.enemyName}을 분석했습니다. 약점: {target.weakPoint}</color>");
        
        // 모든 아군에게 정보 공유
        var allies = FindObjectsOfType<MechCharacter>().Where(m => m != this && m.isAlive).ToList();
        foreach (var ally in allies)
        {
            if (UnityEngine.Random.Range(0f, 1f) < 0.3f) // 30% 확률로 대사
            {
                ally.SayDialogue($"고마워, {mechName}! 약점을 노려보자!", DialogueType.Cooperative);
                break;
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// 방패 재생 능력 - 아군의 방패 HP를 회복 (렉스와의 협력)
    /// </summary>
    /// <param name="rexMech">렉스 기계</param>
    /// <returns>성공하면 true</returns>
    public bool UseShieldRegeneration(RexMech rexMech)
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        // 거리 확인
        float distance = Vector3.Distance(transform.position, rexMech.transform.position);
        if (distance > repairRange) return false;
        
        actionPoints.ConsumeAP(1);
        
        // 방패 HP 회복
        int regenAmount = Mathf.RoundToInt(shieldRegenRate);
        int currentShield = rexMech.currentShieldHP;
        rexMech.currentShieldHP = Mathf.Min(rexMech.shieldMaxHP, currentShield + regenAmount);
        
        int actualRegen = rexMech.currentShieldHP - currentShield;
        
        if (actualRegen > 0)
        {
            SayDialogue($"{rexMech.mechName}, 방패를 강화했어!", DialogueType.Cooperative);
            rexMech.SayDialogue($"고마워, {mechName}! 방패가 더 튼튼해졌어!", DialogueType.Grateful);
            
            Debug.Log($"<color=cyan>{mechName}이 {rexMech.mechName}의 방패를 {actualRegen} 재생했습니다!</color>");
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// EMP 폭탄 - 범위 내 모든 적을 일시 마비
    /// </summary>
    /// <param name="targetPosition">폭발 위치</param>
    /// <param name="radius">폭발 반경</param>
    /// <returns>영향받은 적의 수</returns>
    public int UseEMPBomb(Vector3 targetPosition, float radius = 2f)
    {
        if (!actionPoints.CanUseAP(3)) return 0; // 강력한 능력이므로 3 AP 소모
        
        actionPoints.ConsumeAP(3);
        
        var enemies = FindObjectsOfType<EnemyAI>();
        int affected = 0;
        
        foreach (var enemy in enemies)
        {
            if (enemy.isAlive)
            {
                float distance = Vector3.Distance(enemy.transform.position, targetPosition);
                if (distance <= radius)
                {
                    enemy.ApplySuppression(2f); // 2턴 동안 억제
                    affected++;
                }
            }
        }
        
        if (affected > 0)
        {
            SayDialogue("EMP 폭탄 투하! 모든 적 시스템 다운!", DialogueType.Cooperative);
            Debug.Log($"<color=blue>{mechName}의 EMP 폭탄이 {affected}명의 적을 억제했습니다!</color>");
        }
        
        return affected;
    }
    
    /// <summary>
    /// 자동 수리 드론 배치 - 일정 시간 동안 주변 아군을 자동으로 치료
    /// </summary>
    /// <returns>성공하면 true</returns>
    public bool DeployAutoRepairDrone()
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        actionPoints.ConsumeAP(2);
        
        StartCoroutine(AutoRepairDroneRoutine());
        
        SayDialogue("자동 수리 드론 배치! 모두 치료해줄게!", DialogueType.Cooperative);
        Debug.Log($"<color=green>{mechName}이 자동 수리 드론을 배치했습니다!</color>");
        
        return true;
    }
    
    /// <summary>
    /// 자동 수리 드론 동작 코루틴
    /// </summary>
    private IEnumerator AutoRepairDroneRoutine()
    {
        int duration = 3; // 3턴 동안 지속
        
        for (int turn = 0; turn < duration; turn++)
        {
            yield return new WaitForSeconds(1f); // 턴 간격
            
            // 가장 부상이 심한 아군을 찾아서 치료
            var allies = FindObjectsOfType<MechCharacter>().Where(m => m.isAlive).OrderBy(m => m.stats.currentHP).ToList();
            
            if (allies.Count > 0)
            {
                var targetAlly = allies.First();
                float distance = Vector3.Distance(transform.position, targetAlly.transform.position);
                
                if (distance <= repairRange * 1.5f) // 드론은 더 넓은 범위
                {
                    int droneHeal = Mathf.RoundToInt(repairPower * 0.4f); // 40% 효율
                    targetAlly.Heal(droneHeal);
                    
                    Debug.Log($"<color=green>자동 수리 드론이 {targetAlly.mechName}을 {droneHeal} 치료했습니다.</color>");
                }
            }
        }
        
        Debug.Log("자동 수리 드론 임무 완료.");
    }
    
    /// <summary>
    /// 협상 시도 - 일부 지능형 적과 대화를 통해 전투 회피
    /// </summary>
    /// <param name="target">협상할 적</param>
    /// <returns>성공하면 true</returns>
    public bool AttemptNegotiation(EnemyAI target)
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        if (!target.CanBeNegotiated())
        {
            Debug.LogWarning($"{target.enemyName}은 협상이 불가능합니다.");
            return false;
        }
        
        actionPoints.ConsumeAP(2);
        
        // 협상 성공률 (분석된 적일수록 성공률 높음)
        float successRate = 0.4f;
        if (analyzedEnemies.Contains(target))
        {
            successRate += 0.3f; // 분석된 적이면 +30%
        }
        
        bool success = UnityEngine.Random.Range(0f, 1f) < successRate;
        
        if (success)
        {
            target.ConvertToAlly();
            SayDialogue($"{target.enemyName}과 협상에 성공했어! 이제 우리 편이야!", DialogueType.Cooperative);
            
            Debug.Log($"<color=green>{mechName}이 {target.enemyName}과의 협상에 성공했습니다!</color>");
            return true;
        }
        else
        {
            SayDialogue($"{target.enemyName}... 협상이 결렬됐어.", DialogueType.Damage);
            Debug.Log($"<color=red>{mechName}의 {target.enemyName}과의 협상이 실패했습니다.</color>");
            return false;
        }
    }
    
    /// <summary>
    /// 루나는 다른 기계들보다 피해를 더 많이 받습니다 (낮은 방어력)
    /// </summary>
    public override int TakeDamage(int damage, BodyPartType? targetPart = null)
    {
        // 루나 특성: 피해를 10% 더 받음 (취약한 서포터)
        damage = Mathf.RoundToInt(damage * 1.1f);
        
        int actualDamage = base.TakeDamage(damage, targetPart);
        
        // 피해를 받으면 주변 아군에게 도움 요청 (30% 확률)
        if (actualDamage > 0 && UnityEngine.Random.Range(0f, 1f) < 0.3f)
        {
            var allies = FindObjectsOfType<MechCharacter>().Where(m => m != this && m.isAlive).ToList();
            if (allies.Count > 0)
            {
                var nearestAlly = allies.OrderBy(a => Vector3.Distance(transform.position, a.transform.position)).First();
                nearestAlly.SayDialogue($"{mechName}! 괜찮아?", DialogueType.Worried);
            }
        }
        
        return actualDamage;
    }
    
    /// <summary>
    /// 상태 정보를 문자열로 반환
    /// </summary>
    public override string ToString()
    {
        string status = $"{mechName} (HP: {stats.currentHP}/{stats.maxHP}, AP: {actionPoints.currentAP}/{actionPoints.maxAP})";
        
        if (repairCooldown > 0)
        {
            status += $"\n나노 리페어 쿨다운: {repairCooldown:F1}초";
        }
        
        if (hackCooldown > 0)
        {
            status += $"\n해킹 쿨다운: {hackCooldown:F1}초";
        }
        
        if (analyzedEnemies.Count > 0)
        {
            status += $"\n분석된 적: {analyzedEnemies.Count}마리";
        }
        
        return status;
    }
}

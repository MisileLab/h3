using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

/// <summary>
/// 제로 (Zero) - 스피드 스카우트 (정찰/기동)
/// 핵심 능력: 블리츠 - 적의 경계를 무시하고 빠르게 이동하여 연속 공격
/// 전투 철학: "빨리 끝내고 집에 가자!"
/// </summary>
public class ZeroMech : MechCharacter
{
    [Header("제로 전용 설정")]
    public int blitzAttacks = 3;                    // 블리츠 공격 횟수
    public float blitzDamageMultiplier = 0.7f;      // 블리츠 공격 피해 배율
    public float blitzRange = 6f;                   // 블리츠 사거리
    public float blitzCooldown = 0f;                // 블리츠 쿨다운
    public float blitzMaxCooldown = 3f;             // 블리츠 최대 쿨다운
    
    [Header("정찰 능력")]
    public float scoutRange = 7f;                   // 정찰 사거리
    public int maxStealthTurns = 3;                 // 최대 은신 턴
    public int currentStealthTurns = 0;             // 현재 은신 턴
    // 은신 상태는 부모 클래스 MechCharacter의 isInStealth 사용
    public float stealthDetectionChance = 0.1f;     // 은신 중 발각 확률
    
    [Header("기동성")]
    public int extraMovementRange = 2;              // 추가 이동 거리
    public float dodgeChance = 0.4f;                // 회피 확률
    public int comboAttackCount = 0;                // 연속 공격 카운터
    public float comboResetTime = 2f;               // 콤보 리셋 시간
    private float lastAttackTime = 0f;              // 마지막 공격 시간
    
    [Header("정보 수집")]
    public List<Vector3> scoutedPositions;          // 정찰한 위치들
    public List<EnemyAI> spottedennemies;           // 발견한 적들
    
    protected override void Start()
    {
        // 기본 정보 설정
        mechName = "제로";
        mechType = MechType.Zero;
        
        // 제로의 스탯 (속도 특화)
        stats = new MechStats
        {
            maxHP = 90,         // 중간 HP
            currentHP = 90,
            attack = 22,        // 높은 공격력
            defense = 12,       // 낮은 방어력
            speed = 18,         // 최고 속도
            accuracy = 80,      // 높은 명중률
            evasion = 35        // 최고 회피율
        };
        
        scoutedPositions = new List<Vector3>();
        spottedennemies = new List<EnemyAI>();
        
        base.Start();
    }
    
    protected override void InitializeDialogues()
    {
        dialogueLines = new List<DialogueLine>
        {
            // 턴 시작
            new DialogueLine { type = DialogueType.TurnStart, text = "빨리 끝내고 집에 가자!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "스피드로 승부다!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "적의 움직임을 파악했어!", weight = 0.8f },
            
            // 피해 받을 때
            new DialogueLine { type = DialogueType.Damage, text = "빨라서 다행이야!", weight = 1f },
            new DialogueLine { type = DialogueType.Damage, text = "이런, 맞았네!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "속도가... 떨어지고 있어!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "도움이... 필요해!", weight = 1f },
            
            // 치료 받을 때
            new DialogueLine { type = DialogueType.Healed, text = "고마워! 다시 달릴 수 있겠어!", weight = 1f },
            new DialogueLine { type = DialogueType.Healed, text = "시스템 복구! 속도 업!", weight = 1f },
            
            // 블리츠 공격
            new DialogueLine { type = DialogueType.Cooperative, text = "블리츠 모드 활성화!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "순식간에 끝내버릴게!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "따라올 수 있으면 따라와 봐!", weight = 1f },
            
            // 정찰 성공
            new DialogueLine { type = DialogueType.Cooperative, text = "적의 위치를 파악했어!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "정찰 완료! 데이터 전송 중!", weight = 1f },
            
            // 은신 시
            new DialogueLine { type = DialogueType.Cooperative, text = "은신 모드... 조용히 가자.", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "그림자처럼 움직일게.", weight = 1f },
            
            // 부위 파괴 (특히 다리)
            new DialogueLine { type = DialogueType.PartDestroyed, text = "이동 시스템이...!", weight = 1f },
            
            // 전투 불능
            new DialogueLine { type = DialogueType.Incapacitated, text = "더 이상... 달릴 수 없어...", weight = 1f },
            new DialogueLine { type = DialogueType.Incapacitated, text = "미안... 늦어서...", weight = 1f },
            
            // 회복
            new DialogueLine { type = DialogueType.Revived, text = "다시 달릴 준비 완료!", weight = 1f },
            new DialogueLine { type = DialogueType.Revived, text = "엔진 재가동! 가보자!", weight = 1f },
            
            // 승리
            new DialogueLine { type = DialogueType.Victory, text = "빠른 승부였어! 이제 집에 가자!", weight = 1f },
            new DialogueLine { type = DialogueType.Victory, text = "역시 속도가 생명이야!", weight = 1f }
        };
    }
    
    public override void StartTurn()
    {
        base.StartTurn();
        
        // 은신 지속 시간 감소
        if (isInStealth && currentStealthTurns > 0)
        {
            currentStealthTurns--;
            if (currentStealthTurns <= 0)
            {
                ExitStealth();
            }
        }
        
        // 콤보 리셋 체크
        if (Time.time - lastAttackTime > comboResetTime)
        {
            comboAttackCount = 0;
        }
    }
    
    public override void UpdateCooldowns(float deltaTime)
    {
        base.UpdateCooldowns(deltaTime);
        
        // 블리츠 쿨다운 업데이트
        if (blitzCooldown > 0)
        {
            blitzCooldown -= deltaTime;
            if (blitzCooldown <= 0)
            {
                blitzCooldown = 0;
                Debug.Log($"{mechName}의 블리츠 쿨다운이 완료되었습니다.");
            }
        }
    }
    
    /// <summary>
    /// 블리츠 공격 - 적의 경계를 무시하고 빠르게 이동하여 연속 공격
    /// </summary>
    /// <param name="targets">공격할 대상들</param>
    /// <returns>성공하면 true</returns>
    public bool UseBlitzAttack(List<EnemyAI> targets)
    {
        // AP 및 쿨다운 확인
        if (!actionPoints.CanUseAP(3))
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 3)");
            return false;
        }
        
        if (blitzCooldown > 0)
        {
            Debug.LogWarning($"블리츠가 쿨다운 중입니다. ({blitzCooldown:F1}초 남음)");
            return false;
        }
        
        if (targets == null || targets.Count == 0)
        {
            Debug.LogWarning("공격할 대상이 없습니다.");
            return false;
        }
        
        // 사거리 내의 대상만 필터링
        var validTargets = targets.Where(t => t.isAlive && 
            Vector3.Distance(transform.position, t.transform.position) <= blitzRange).ToList();
        
        if (validTargets.Count == 0)
        {
            Debug.LogWarning("사거리 내에 유효한 대상이 없습니다.");
            return false;
        }
        
        // AP 소모
        actionPoints.ConsumeAP(3);
        
        // 블리츠 공격 실행
        StartCoroutine(BlitzAttackSequence(validTargets));
        
        // 쿨다운 설정
        blitzCooldown = blitzMaxCooldown;
        
        return true;
    }
    
    /// <summary>
    /// 블리츠 공격 시퀀스 코루틴
    /// </summary>
    /// <param name="targets">공격 대상들</param>
    private IEnumerator BlitzAttackSequence(List<EnemyAI> targets)
    {
        SayDialogue("블리츠 모드 활성화!", DialogueType.Cooperative);
        
        Vector3 originalPosition = transform.position;
        int attacksPerformed = 0;
        int maxAttacks = Mathf.Min(blitzAttacks, targets.Count);
        
        for (int i = 0; i < maxAttacks; i++)
        {
            var target = targets[i];
            if (!target.isAlive) continue;
            
            // 대상에게 순간이동
            transform.position = target.transform.position + Vector3.right * 0.5f;
            
            // 공격
            int damage = Mathf.RoundToInt(stats.attack * blitzDamageMultiplier);
            target.TakeDamage(damage);
            
            attacksPerformed++;
            comboAttackCount++;
            lastAttackTime = Time.time;
            
            Debug.Log($"<color=yellow>블리츠 공격 {i + 1}: {target.enemyName}에게 {damage} 피해!</color>");
            
            yield return new WaitForSeconds(0.2f); // 짧은 딜레이
        }
        
        // 원래 위치로 복귀
        transform.position = originalPosition;
        
        // 마무리 대사
        if (attacksPerformed >= 3)
        {
            SayDialogue("완벽한 콤보!", DialogueType.Cooperative);
        }
        else if (attacksPerformed >= 2)
        {
            SayDialogue("나쁘지 않아!", DialogueType.Cooperative);
        }
        
        Debug.Log($"<color=yellow>블리츠 공격 완료: {attacksPerformed}번 공격, 콤보 {comboAttackCount}</color>");
    }
    
    /// <summary>
    /// 정찰 능력 - 넓은 범위의 적과 지형 정보를 수집
    /// </summary>
    /// <param name="targetPosition">정찰할 위치</param>
    /// <returns>발견한 적의 수</returns>
    public int UseScout(Vector3 targetPosition)
    {
        if (!actionPoints.CanUseAP(1)) return 0;
        
        // 이동 거리 제한 확인
        float distance = Vector3.Distance(transform.position, targetPosition);
        float maxScoutDistance = scoutRange;
        
        if (distance > maxScoutDistance)
        {
            Debug.LogWarning($"정찰 거리가 너무 멀어요. (거리: {distance:F1}, 최대: {maxScoutDistance})");
            return 0;
        }
        
        actionPoints.ConsumeAP(1);
        
        // 위치 이동 (실제로는 시각적 효과만)
        Vector3 originalPos = transform.position;
        
        // 주변 적들 탐지
        var enemies = FindObjectsOfType<EnemyAI>();
        var discoveredEnemies = new List<EnemyAI>();
        
        foreach (var enemy in enemies)
        {
            if (enemy.isAlive)
            {
                float distanceToTarget = Vector3.Distance(targetPosition, enemy.transform.position);
                if (distanceToTarget <= scoutRange * 0.5f) // 정찰 지점 기준으로 절반 범위
                {
                    if (!spottedennemies.Contains(enemy))
                    {
                        spottedennemies.Add(enemy);
                        discoveredEnemies.Add(enemy);
                        
                        // 적 정보 일부 노출
                        enemy.RevealWeakness();
                    }
                }
            }
        }
        
        // 정찰한 위치 기록
        if (!scoutedPositions.Contains(targetPosition))
        {
            scoutedPositions.Add(targetPosition);
        }
        
        if (discoveredEnemies.Count > 0)
        {
            SayDialogue($"{discoveredEnemies.Count}마리의 적을 발견했어!", DialogueType.Cooperative);
            
            // 동료들에게 정보 공유
            var allies = FindObjectsOfType<MechCharacter>().Where(m => m != this && m.isAlive).ToList();
            foreach (var ally in allies)
            {
                if (UnityEngine.Random.Range(0f, 1f) < 0.4f) // 40% 확률로 대사
                {
                    ally.SayDialogue($"고마워, {mechName}! 좋은 정보야!", DialogueType.Cooperative);
                    break;
                }
            }
        }
        else
        {
            SayDialogue("이 지역은 안전해!", DialogueType.Cooperative);
        }
        
        Debug.Log($"<color=cyan>{mechName}이 정찰을 완료했습니다. 발견한 적: {discoveredEnemies.Count}마리</color>");
        return discoveredEnemies.Count;
    }
    
    /// <summary>
    /// 은신 능력 - 일정 턴 동안 적의 공격을 받지 않음
    /// </summary>
    /// <param name="duration">은신 지속 턴</param>
    /// <returns>성공하면 true</returns>
    public bool UseStealth(int duration = 2)
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        if (isInStealth)
        {
            Debug.LogWarning("이미 은신 상태입니다.");
            return false;
        }
        
        actionPoints.ConsumeAP(2);
        
        // 은신 활성화
        isInStealth = true;
        currentStealthTurns = Mathf.Min(duration, maxStealthTurns);
        
        SayDialogue("은신 모드... 조용히 가자.", DialogueType.Cooperative);
        
        Debug.Log($"<color=gray>{mechName}이 {currentStealthTurns}턴 동안 은신합니다.</color>");
        return true;
    }
    
    /// <summary>
    /// 은신 해제
    /// </summary>
    private void ExitStealth()
    {
        isInStealth = false;
        currentStealthTurns = 0;
        
        SayDialogue("은신 해제! 다시 행동 개시!", DialogueType.Cooperative);
        Debug.Log($"<color=gray>{mechName}의 은신이 해제되었습니다.</color>");
    }
    
    /// <summary>
    /// 히트 앤 런 - 공격 후 즉시 안전한 위치로 이동
    /// </summary>
    /// <param name="target">공격 대상</param>
    /// <param name="escapePosition">도망칠 위치</param>
    /// <returns>성공하면 true</returns>
    public bool UseHitAndRun(EnemyAI target, Vector3 escapePosition)
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        float attackDistance = Vector3.Distance(transform.position, target.transform.position);
        float escapeDistance = Vector3.Distance(target.transform.position, escapePosition);
        
        if (attackDistance > 2f || escapeDistance < 2f) // 공격 거리는 가깝고, 도망 거리는 충분히 멀어야 함
        {
            Debug.LogWarning("히트 앤 런 조건이 맞지 않습니다.");
            return false;
        }
        
        actionPoints.ConsumeAP(2);
        
        // 공격
        int damage = stats.attack;
        target.TakeDamage(damage);
        
        // 즉시 이동
        transform.position = escapePosition;
        
        SayDialogue("히트 앤 런! 따라올 수 있으면 따라와 봐!", DialogueType.Cooperative);
        
        Debug.Log($"<color=yellow>{mechName}이 히트 앤 런을 사용했습니다! {target.enemyName}에게 {damage} 피해 후 도주!</color>");
        return true;
    }
    
    /// <summary>
    /// 연막탄 - 범위 내 모든 유닛의 명중률 감소
    /// </summary>
    /// <param name="targetPosition">연막탄 투하 위치</param>
    /// <param name="radius">연막 범위</param>
    /// <returns>영향받은 유닛 수</returns>
    public int UseSmokeBomb(Vector3 targetPosition, float radius = 3f)
    {
        if (!actionPoints.CanUseAP(1)) return 0;
        
        actionPoints.ConsumeAP(1);
        
        // 연막 효과 시뮬레이션 (실제 구현에서는 시각 효과 추가)
        var allUnits = new List<MonoBehaviour>();
        allUnits.AddRange(FindObjectsOfType<MechCharacter>());
        allUnits.AddRange(FindObjectsOfType<EnemyAI>());
        
        int affected = 0;
        foreach (var unit in allUnits)
        {
            float distance = Vector3.Distance(unit.transform.position, targetPosition);
            if (distance <= radius)
            {
                // 명중률 감소 효과 (실제로는 버프/디버프 시스템으로 구현)
                affected++;
            }
        }
        
        SayDialogue("연막탄 투하! 시야를 차단한다!", DialogueType.Cooperative);
        
        Debug.Log($"<color=gray>{mechName}이 연막탄을 사용했습니다. 영향받은 유닛: {affected}명</color>");
        return affected;
    }
    
    /// <summary>
    /// 제로는 높은 회피율로 피해를 회피할 수 있습니다
    /// </summary>
    public override int TakeDamage(int damage, BodyPartType? targetPart = null)
    {
        // 은신 중에는 발각 확률 체크
        if (isInStealth)
        {
            if (UnityEngine.Random.Range(0f, 1f) < stealthDetectionChance)
            {
                ExitStealth();
                SayDialogue("발각됐어! 은신 해제!", DialogueType.Damage);
            }
            else
            {
                Debug.Log($"<color=gray>{mechName}이 은신 상태로 공격을 회피했습니다!</color>");
                return 0; // 은신 중에는 피해 없음
            }
        }
        
        // 회피 확률 체크
        float totalDodgeChance = dodgeChance;
        
        // 콤보 중일 때 회피율 증가
        if (comboAttackCount >= 2)
        {
            totalDodgeChance += 0.1f * comboAttackCount; // 콤보마다 +10% 회피율
        }
        
        if (UnityEngine.Random.Range(0f, 1f) < totalDodgeChance)
        {
            SayDialogue("너무 느려!", DialogueType.Cooperative);
            Debug.Log($"<color=cyan>{mechName}이 공격을 회피했습니다! (회피율: {totalDodgeChance * 100:F1}%)</color>");
            return 0;
        }
        
        // 기본 피해 처리
        int actualDamage = base.TakeDamage(damage, targetPart);
        
        // 콤보 리셋
        if (actualDamage > 0)
        {
            comboAttackCount = 0;
        }
        
        return actualDamage;
    }
    
    /// <summary>
    /// 제로의 특수 이동 - 추가 이동 거리를 가집니다
    /// </summary>
    /// <param name="targetPosition">이동할 위치</param>
    /// <returns>성공하면 true</returns>
    public bool UseSpeedMove(Vector3 targetPosition)
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        float distance = Vector3.Distance(transform.position, targetPosition);
        float maxDistance = extraMovementRange + 2f; // 기본 이동 + 추가 이동
        
        if (distance > maxDistance)
        {
            Debug.LogWarning($"이동 거리가 너무 멉니다. (거리: {distance:F1}, 최대: {maxDistance})");
            return false;
        }
        
        actionPoints.ConsumeAP(1);
        
        transform.position = targetPosition;
        
        SayDialogue("빠른 이동!", DialogueType.Cooperative);
        Debug.Log($"<color=cyan>{mechName}이 고속 이동했습니다! ({distance:F1} 유닛)</color>");
        
        return true;
    }
    
    /// <summary>
    /// 상태 정보를 문자열로 반환
    /// </summary>
    public override string ToString()
    {
        string status = $"{mechName} (HP: {stats.currentHP}/{stats.maxHP}, AP: {actionPoints.currentAP}/{actionPoints.maxAP})";
        
        if (isInStealth)
        {
            status += $"\n은신 상태 ({currentStealthTurns}턴 남음)";
        }
        
        if (blitzCooldown > 0)
        {
            status += $"\n블리츠 쿨다운: {blitzCooldown:F1}초";
        }
        
        if (comboAttackCount > 0)
        {
            status += $"\n콤보: {comboAttackCount}연속";
        }
        
        if (spottedennemies.Count > 0)
        {
            status += $"\n발견한 적: {spottedennemies.Count}마리";
        }
        
        return status;
    }
}

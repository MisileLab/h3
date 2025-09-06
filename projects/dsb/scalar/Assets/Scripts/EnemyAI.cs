using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class EnemyAI : MonoBehaviour
{
    [Header("적 정보")]
    public string enemyName;
    public EnemyType enemyType;
    public int maxHP = 100;
    public int currentHP = 100;
    public int attack = 20;
    public int defense = 10;
    public int speed = 8;
    public int accuracy = 70;
    
    [Header("AI 행동")]
    public float detectionRange = 5f;
    public float attackRange = 2f;
    public bool isHacked = false;
    public bool isSuppressed = false;
    public bool isTaunted = false;
    public MechCharacter tauntedTarget;
    public float tauntDuration = 0f;
    
    [Header("상태")]
    public bool isAlive = true;
    public bool canBeNegotiated = false;
    public bool isConverted = false;
    
    [Header("약점")]
    public bool weaknessRevealed = false;
    public BodyPartType weakPoint = BodyPartType.Torso;
    
    private MechCharacter[] playerMechs;
    private float hackDuration = 0f;
    private float suppressionDuration = 0f;
    
    private void Start()
    {
        playerMechs = FindObjectsOfType<MechCharacter>();
        currentHP = maxHP;
    }
    
    private void Update()
    {
        UpdateStatusEffects();
    }
    
    public void UpdateStatusEffects()
    {
        if (isHacked && hackDuration > 0)
        {
            hackDuration -= Time.deltaTime;
            if (hackDuration <= 0)
            {
                isHacked = false;
                Debug.Log($"{enemyName}의 해킹이 해제되었습니다.");
            }
        }
        
        if (isSuppressed && suppressionDuration > 0)
        {
            suppressionDuration -= Time.deltaTime;
            if (suppressionDuration <= 0)
            {
                isSuppressed = false;
                Debug.Log($"{enemyName}의 억제가 해제되었습니다.");
            }
        }
        
        if (isTaunted && tauntDuration > 0)
        {
            tauntDuration -= Time.deltaTime;
            if (tauntDuration <= 0)
            {
                isTaunted = false;
                tauntedTarget = null;
                Debug.Log($"{enemyName}의 도발이 해제되었습니다.");
            }
        }
    }
    
    public void TakeDamage(float damage)
    {
        if (!isAlive) return;
        
        float actualDamage = damage * (1.0f - defense / 100.0f);
        currentHP = Mathf.Max(0, currentHP - (int)actualDamage);
        
        Debug.Log($"{enemyName}이 {actualDamage}의 피해를 받았습니다. (남은 HP: {currentHP})");
        
        if (currentHP <= 0)
        {
            Die();
        }
    }
    
    private void Die()
    {
        isAlive = false;
        Debug.Log($"{enemyName}이 파괴되었습니다.");
        // 파괴 효과 및 아이템 드롭 로직
    }
    
    public void GetHacked(float duration)
    {
        isHacked = true;
        hackDuration = duration;
        Debug.Log($"{enemyName}이 해킹되었습니다. ({duration}초)");
    }
    
    public void ApplySuppression(float duration)
    {
        isSuppressed = true;
        suppressionDuration = duration;
        Debug.Log($"{enemyName}이 억제되었습니다. ({duration}초)");
    }
    
    public void SetTauntedTarget(MechCharacter target, float duration)
    {
        isTaunted = true;
        tauntedTarget = target;
        tauntDuration = duration;
        Debug.Log($"{enemyName}이 {target.mechName}에게 도발되었습니다.");
    }
    
    public void RevealInfo()
    {
        weaknessRevealed = true;
        Debug.Log($"{enemyName}의 정보가 노출되었습니다. 약점: {weakPoint}");
    }
    
    public void RevealWeakness()
    {
        weaknessRevealed = true;
        Debug.Log($"{enemyName}의 약점이 분석되었습니다: {weakPoint}");
    }
    
    public bool CanBeNegotiated()
    {
        return canBeNegotiated && !isConverted;
    }
    
    public void ConvertToAlly()
    {
        isConverted = true;
        Debug.Log($"{enemyName}이 아군이 되었습니다!");
        // 아군으로 전환하는 로직
    }
    
    public MechCharacter FindWeakestTarget()
    {
        MechCharacter weakest = null;
        int lowestHP = int.MaxValue;
        
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive && mech.stats.currentHP < lowestHP)
            {
                lowestHP = mech.stats.currentHP;
                weakest = mech;
            }
        }
        
        return weakest;
    }
    
    public MechCharacter FindNearestTarget()
    {
        MechCharacter nearest = null;
        float shortestDistance = float.MaxValue;
        
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive)
            {
                float distance = Vector3.Distance(transform.position, mech.transform.position);
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    nearest = mech;
                }
            }
        }
        
        return nearest;
    }
    
    public bool PerformAction()
    {
        if (!isAlive || isHacked) return false;
        
        MechCharacter target = null;
        bool actionPerformed = false;
        
        // 도발된 경우 도발한 대상만 공격
        if (isTaunted && tauntedTarget != null && tauntedTarget.isAlive)
        {
            target = tauntedTarget;
        }
        else
        {
            // AI 타입에 따른 행동 패턴
            switch (enemyType)
            {
                case EnemyType.Scrapper:
                    actionPerformed = PerformScrapperAction();
                    break;
                case EnemyType.Sentinel:
                    actionPerformed = PerformSentinelAction();
                    break;
                case EnemyType.Interceptor:
                    actionPerformed = PerformInterceptorAction();
                    break;
                case EnemyType.Enforcer:
                    actionPerformed = PerformEnforcerAction();
                    break;
            }
        }
        
        // 도발된 경우 단순 공격
        if (target != null && !actionPerformed)
        {
            float distance = Vector3.Distance(transform.position, target.transform.position);
            
            if (distance <= attackRange)
            {
                actionPerformed = Attack(target);
            }
            else if (distance <= detectionRange)
            {
                actionPerformed = MoveTowardsTarget(target);
            }
        }
        
        return actionPerformed;
    }
    
    /// <summary>
    /// 스크래퍼의 행동 패턴 - 예측 불가능한 행동
    /// </summary>
    private bool PerformScrapperAction()
    {
        // 랜덤하게 행동 결정
        int action = Random.Range(0, 4);
        
        switch (action)
        {
            case 0: // 가장 약한 적 공격
                var weakTarget = FindWeakestTarget();
                if (weakTarget != null)
                {
                    return AttackOrMoveToTarget(weakTarget);
                }
                break;
                
            case 1: // 랜덤 대상 공격
                var randomTarget = FindRandomTarget();
                if (randomTarget != null)
                {
                    return AttackOrMoveToTarget(randomTarget);
                }
                break;
                
            case 2: // 자폭 시도 (30% 확률, HP가 낮을 때)
                if (currentHP < maxHP * 0.3f && Random.Range(0f, 1f) < 0.3f)
                {
                    return AttemptSelfDestruct();
                }
                break;
                
            case 3: // 혼란 공격 (아군 오사 가능)
                return AttemptConfusedAttack();
        }
        
        return false;
    }
    
    /// <summary>
    /// 센티넬의 행동 패턴 - 특정 영역을 수호
    /// </summary>
    private bool PerformSentinelAction()
    {
        Vector3 guardPosition = transform.position; // 현재 위치를 경계 지점으로 가정
        
        // 경계 범위 내 침입자 확인
        var intruders = FindTargetsInRange(guardPosition, detectionRange);
        
        if (intruders.Count > 0)
        {
            // 가장 가까운 침입자 공격
            var nearestIntruder = intruders.OrderBy(t => Vector3.Distance(transform.position, t.transform.position)).First();
            return AttackOrMoveToTarget(nearestIntruder);
        }
        else
        {
            // 침입자가 없으면 경계 위치로 복귀 또는 대기
            Debug.Log($"{enemyName}이 경계를 서고 있습니다.");
            return true;
        }
    }
    
    /// <summary>
    /// 인터셉터의 행동 패턴 - 고속 기동, 히트 앤 런
    /// </summary>
    private bool PerformInterceptorAction()
    {
        // 우선 타겟: 루나 같은 약한 서포터
        var priorityTarget = FindWeakestTarget();
        
        if (priorityTarget != null)
        {
            float distance = Vector3.Distance(transform.position, priorityTarget.transform.position);
            
            if (distance <= attackRange)
            {
                // 공격 후 즉시 후퇴
                bool attackSuccess = Attack(priorityTarget);
                if (attackSuccess)
                {
                    // 안전한 위치로 이동
                    Vector3 retreatPosition = FindSafeRetreatPosition();
                    if (retreatPosition != Vector3.zero)
                    {
                        transform.position = retreatPosition;
                        Debug.Log($"{enemyName}이 히트 앤 런 전술을 사용했습니다!");
                    }
                }
                return attackSuccess;
            }
            else
            {
                // 고속 접근
                return MoveTowardsTarget(priorityTarget, 1.5f); // 1.5배 속도
            }
        }
        
        return false;
    }
    
    /// <summary>
    /// 엔포서의 행동 패턴 - 압도적인 화력과 방어력
    /// </summary>
    private bool PerformEnforcerAction()
    {
        // 협력 행동을 방해하는 스킬 사용 (50% 확률)
        if (Random.Range(0f, 1f) < 0.5f && TryDisruptCooperation())
        {
            return true;
        }
        
        // 가장 위협적인 대상 공격 (높은 공격력 또는 HP가 높은 대상)
        var mostThreatening = FindMostThreateningTarget();
        if (mostThreatening != null)
        {
            return AttackOrMoveToTarget(mostThreatening);
        }
        
        // 광역 공격 시도 (30% 확률)
        if (Random.Range(0f, 1f) < 0.3f)
        {
            return AttemptAreaAttack();
        }
        
        return false;
    }
    
    public void StartTurn()
    {
        // 적의 턴 시작 처리
    }
    
    public void EndTurn()
    {
        // 적의 턴 종료 처리
    }
    
    
    private bool MoveTowardsTarget(MechCharacter target, float speedMultiplier = 1f)
    {
        if (target == null) return false;
        
        Vector3 direction = (target.transform.position - transform.position).normalized;
        transform.position += direction * speed * speedMultiplier * Time.deltaTime;
        Debug.Log($"{enemyName}이 {target.mechName}을 향해 이동합니다.");
        return true;
    }
    
    /// <summary>
    /// 대상을 공격하거나 접근합니다
    /// </summary>
    private bool AttackOrMoveToTarget(MechCharacter target)
    {
        if (target == null) return false;
        
        float distance = Vector3.Distance(transform.position, target.transform.position);
        
        if (distance <= attackRange)
        {
            return Attack(target);
        }
        else if (distance <= detectionRange)
        {
            return MoveTowardsTarget(target);
        }
        
        return false;
    }
    
    /// <summary>
    /// 랜덤한 타겟을 찾습니다
    /// </summary>
    private MechCharacter FindRandomTarget()
    {
        var aliveTargets = playerMechs.Where(m => m.isAlive).ToArray();
        if (aliveTargets.Length == 0) return null;
        
        return aliveTargets[Random.Range(0, aliveTargets.Length)];
    }
    
    /// <summary>
    /// 특정 위치 주변의 타겟들을 찾습니다
    /// </summary>
    private List<MechCharacter> FindTargetsInRange(Vector3 position, float range)
    {
        return playerMechs.Where(m => m.isAlive && 
            Vector3.Distance(m.transform.position, position) <= range).ToList();
    }
    
    /// <summary>
    /// 가장 위협적인 타겟을 찾습니다 (공격력이 높거나 HP가 많은 대상)
    /// </summary>
    private MechCharacter FindMostThreateningTarget()
    {
        var aliveTargets = playerMechs.Where(m => m.isAlive).ToList();
        if (aliveTargets.Count == 0) return null;
        
        return aliveTargets.OrderByDescending(m => m.stats.attack + m.stats.currentHP).First();
    }
    
    /// <summary>
    /// 안전한 후퇴 위치를 찾습니다
    /// </summary>
    private Vector3 FindSafeRetreatPosition()
    {
        // 현재 위치에서 가장 가까운 플레이어와 반대 방향으로 이동
        var nearestPlayer = FindNearestTarget();
        if (nearestPlayer != null)
        {
            Vector3 retreatDirection = (transform.position - nearestPlayer.transform.position).normalized;
            return transform.position + retreatDirection * 2f; // 2칸 뒤로 후퇴
        }
        
        return Vector3.zero;
    }
    
    /// <summary>
    /// 자폭을 시도합니다
    /// </summary>
    private bool AttemptSelfDestruct()
    {
        // 주변 모든 유닛에게 피해
        var nearbyTargets = FindTargetsInRange(transform.position, 2f);
        
        if (nearbyTargets.Count > 0)
        {
            int selfDestructDamage = attack * 2; // 공격력의 2배
            
            foreach (var target in nearbyTargets)
            {
                target.TakeDamage(selfDestructDamage);
                Debug.Log($"{enemyName}의 자폭으로 {target.mechName}이 {selfDestructDamage} 피해!");
            }
            
            Debug.Log($"<color=red>{enemyName}이 자폭했습니다!</color>");
            Die();
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// 혼란 공격을 시도합니다 (아군 오사 가능)
    /// </summary>
    private bool AttemptConfusedAttack()
    {
        // 50% 확률로 잘못된 대상 공격
        if (Random.Range(0f, 1f) < 0.5f)
        {
            var allEnemies = FindObjectsOfType<EnemyAI>().Where(e => e != this && e.isAlive).ToList();
            if (allEnemies.Count > 0)
            {
                var friendlyFire = allEnemies[Random.Range(0, allEnemies.Count)];
                friendlyFire.TakeDamage(attack / 2); // 절반 피해
                Debug.Log($"<color=yellow>{enemyName}이 혼란 상태에서 {friendlyFire.enemyName}을 공격했습니다!</color>");
                return true;
            }
        }
        
        // 일반 공격
        var target = FindRandomTarget();
        return target != null ? Attack(target) : false;
    }
    
    /// <summary>
    /// 협력을 방해하는 스킬을 사용합니다
    /// </summary>
    private bool TryDisruptCooperation()
    {
        // 가장 가까운 두 플레이어 사이에 방해 공격
        var players = playerMechs.Where(m => m.isAlive).ToList();
        
        for (int i = 0; i < players.Count - 1; i++)
        {
            for (int j = i + 1; j < players.Count; j++)
            {
                float distance = Vector3.Distance(players[i].transform.position, players[j].transform.position);
                if (distance <= 2f) // 협력 거리 내에 있으면
                {
                    // 방해 공격 (둘 다 약간의 피해)
                    players[i].TakeDamage(attack / 3);
                    players[j].TakeDamage(attack / 3);
                    
                    Debug.Log($"<color=orange>{enemyName}이 {players[i].mechName}과 {players[j].mechName}의 협력을 방해했습니다!</color>");
                    return true;
                }
            }
        }
        
        return false;
    }
    
    /// <summary>
    /// 광역 공격을 시도합니다
    /// </summary>
    private bool AttemptAreaAttack()
    {
        Vector3 targetCenter = CalculateBestAreaAttackPosition();
        
        if (targetCenter != Vector3.zero)
        {
            var targetsInArea = FindTargetsInRange(targetCenter, 2f);
            
            if (targetsInArea.Count >= 2) // 2명 이상이 맞을 수 있을 때만
            {
                foreach (var target in targetsInArea)
                {
                    int areaDamage = Mathf.RoundToInt(attack * 0.8f); // 80% 위력
                    target.TakeDamage(areaDamage);
                }
                
                Debug.Log($"<color=orange>{enemyName}이 광역 공격으로 {targetsInArea.Count}명을 공격했습니다!</color>");
                return true;
            }
        }
        
        return false;
    }
    
    /// <summary>
    /// 최적의 광역 공격 위치를 계산합니다
    /// </summary>
    private Vector3 CalculateBestAreaAttackPosition()
    {
        var players = playerMechs.Where(m => m.isAlive).ToList();
        Vector3 bestPosition = Vector3.zero;
        int maxTargets = 0;
        
        foreach (var player in players)
        {
            var nearbyTargets = FindTargetsInRange(player.transform.position, 2f);
            if (nearbyTargets.Count > maxTargets)
            {
                maxTargets = nearbyTargets.Count;
                bestPosition = player.transform.position;
            }
        }
        
        return bestPosition;
    }
    
    /// <summary>
    /// 공격 메서드 개선 (반환값 추가)
    /// </summary>
    private bool Attack(MechCharacter target)
    {
        if (isSuppressed)
        {
            Debug.Log($"{enemyName}이 억제되어 공격할 수 없습니다.");
            return false;
        }
        
        if (target == null || !target.isAlive) return false;
        
        float hitChance = accuracy / 100.0f;
        bool hit = Random.Range(0f, 1f) < hitChance;
        
        if (hit)
        {
            target.TakeDamage(attack);
            Debug.Log($"<color=red>{enemyName}이 {target.mechName}을 공격했습니다! ({attack} 피해)</color>");
            return true;
        }
        else
        {
            Debug.Log($"{enemyName}의 공격이 빗나갔습니다.");
            return false;
        }
    }
}

public enum EnemyType
{
    Scrapper,    // 예측 불가능한 행동
    Sentinel,    // 특정 영역 수호
    Interceptor, // 고속 기동, 히트 앤 런
    Enforcer     // 압도적인 화력과 방어력
}

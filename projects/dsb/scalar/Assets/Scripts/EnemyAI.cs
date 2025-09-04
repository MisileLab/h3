using UnityEngine;
using System.Collections;

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
    
    private void UpdateStatusEffects()
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
    
    public void PerformAction()
    {
        if (!isAlive || isHacked) return;
        
        MechCharacter target = null;
        
        // 도발된 경우 도발한 대상만 공격
        if (isTaunted && tauntedTarget != null && tauntedTarget.isAlive)
        {
            target = tauntedTarget;
        }
        else
        {
            // AI 타입에 따른 타겟 선택
            switch (enemyType)
            {
                case EnemyType.Scrapper:
                    target = FindWeakestTarget();
                    break;
                case EnemyType.Sentinel:
                    target = FindNearestTarget();
                    break;
                case EnemyType.Interceptor:
                    target = FindWeakestTarget(); // 루나 같은 약한 대상 우선
                    break;
                case EnemyType.Enforcer:
                    target = FindNearestTarget();
                    break;
            }
        }
        
        if (target != null)
        {
            float distance = Vector3.Distance(transform.position, target.transform.position);
            
            if (distance <= attackRange)
            {
                Attack(target);
            }
            else if (distance <= detectionRange)
            {
                MoveTowardsTarget(target);
            }
        }
    }
    
    public void StartTurn()
    {
        // 적의 턴 시작 처리
    }
    
    public void EndTurn()
    {
        // 적의 턴 종료 처리
    }
    
    private void Attack(MechCharacter target)
    {
        if (isSuppressed)
        {
            Debug.Log($"{enemyName}이 억제되어 공격할 수 없습니다.");
            return;
        }
        
        float hitChance = accuracy / 100.0f;
        bool hit = Random.Range(0f, 1f) < hitChance;
        
        if (hit)
        {
            target.TakeDamage(attack);
            Debug.Log($"{enemyName}이 {target.mechName}을 공격했습니다!");
        }
        else
        {
            Debug.Log($"{enemyName}의 공격이 빗나갔습니다.");
        }
    }
    
    private void MoveTowardsTarget(MechCharacter target)
    {
        Vector3 direction = (target.transform.position - transform.position).normalized;
        transform.position += direction * speed * Time.deltaTime;
        Debug.Log($"{enemyName}이 {target.mechName}을 향해 이동합니다.");
    }
}

public enum EnemyType
{
    Scrapper,    // 예측 불가능한 행동
    Sentinel,    // 특정 영역 수호
    Interceptor, // 고속 기동, 히트 앤 런
    Enforcer     // 압도적인 화력과 방어력
}

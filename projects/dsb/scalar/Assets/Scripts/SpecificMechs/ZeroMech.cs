using UnityEngine;

public class ZeroMech : MechCharacter
{
    [Header("제로 특수 능력")]
    public float blitzDamage = 40f;
    public float stealthDuration = 3f;
    public int stealthRange = 2;
    
    private void Start()
    {
        mechType = MechType.Zero;
        mechName = "제로";
        
        // 제로 스탯 설정
        stats.maxHP = 80;
        stats.currentHP = 80;
        stats.maxAP = 5;
        stats.currentAP = 5;
        stats.speed = 20;
        stats.attack = 30;
        stats.defense = 8;
        stats.accuracy = 85;
        stats.evasion = 40;
        stats.stealthSkill = 30;
        
        InitializeBodyParts();
        InitializeTrustLevels();
        InitializeCoopSkills();
    }
    
    public void BlitzAttack(EnemyAI target)
    {
        if (!CanUseSkill("Blitz") || stats.currentAP < 2) return;
        
        // 블리츠: 적의 경계를 무시하고 빠르게 이동하여 연속 공격
        Vector3 originalPosition = transform.position;
        Vector3 targetPosition = target.transform.position;
        
        // 빠른 이동
        transform.position = Vector3.Lerp(originalPosition, targetPosition, 0.8f);
        
        // 연속 공격
        StartCoroutine(PerformBlitzAttacks(target, 3));
        
        UseSkill("Blitz", 4f);
        ConsumeAP(2);
        
        TriggerDialogue("블리츠", "빨리 끝내고 집에 가자!");
    }
    
    public void StealthMode()
    {
        if (!CanUseSkill("Stealth") || stats.currentAP < 1) return;
        
        // 은신 모드 활성화
        isInStealth = true;
        stats.evasion += 30;
        UseSkill("Stealth", 3f);
        ConsumeAP(1);
        
        TriggerDialogue("은신", "조용히 움직일게.");
        
        // 일정 시간 후 은신 해제
        StartCoroutine(RemoveStealthAfterTime(stealthDuration));
    }
    
    public void TacticalSwap(MechCharacter ally)
    {
        if (!CanUseSkill("TacticalSwap") || stats.currentAP < 1) return;
        
        // 전술 이동: 인접한 아군과 위치 교환
        float distance = Vector3.Distance(transform.position, ally.transform.position);
        if (distance <= 2f)
        {
            Vector3 myPosition = transform.position;
            Vector3 allyPosition = ally.transform.position;
            
            transform.position = allyPosition;
            ally.transform.position = myPosition;
            
            UseSkill("TacticalSwap", 2f);
            ConsumeAP(1);
            
            TriggerDialogue("전술 이동", "위치 바꿔!");
            IncreaseTrust(ally.mechType, 3);
        }
        else
        {
            TriggerDialogue("거리 부족", "너무 멀어서 바꿀 수 없어.");
        }
    }
    
    public void Reconnaissance()
    {
        if (!CanUseSkill("Recon") || stats.currentAP < 1) return;
        
        // 정찰: 주변 적들의 정보를 파악
        Collider[] enemies = Physics.OverlapSphere(transform.position, 8f);
        
        foreach (Collider enemy in enemies)
        {
            EnemyAI enemyAI = enemy.GetComponent<EnemyAI>();
            if (enemyAI != null)
            {
                enemyAI.RevealInfo();
            }
        }
        
        UseSkill("Recon", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("정찰", "적의 위치를 파악했어!");
    }
    
    public void EvasiveManeuver()
    {
        if (!CanUseSkill("EvasiveManeuver") || stats.currentAP < 1) return;
        
        // 회피 기동: 다음 공격을 완전히 회피
        stats.evasion += 50;
        UseSkill("EvasiveManeuver", 1f);
        ConsumeAP(1);
        
        TriggerDialogue("회피 기동", "이건 피할 수 있어!");
        
        // 1턴 후 효과 제거
        StartCoroutine(RemoveEvasiveAfterTime(1f));
    }
    
    public void QuickStrike(EnemyAI target)
    {
        if (!CanUseSkill("QuickStrike") || stats.currentAP < 1) return;
        
        // 빠른 일격: 낮은 피해지만 AP 소모 없이 추가 공격 가능
        float damage = stats.attack * 0.7f;
        target.TakeDamage(damage);
        
        UseSkill("QuickStrike", 1f);
        ConsumeAP(1);
        
        TriggerDialogue("빠른 일격", "한 방 더!");
    }
    
    private System.Collections.IEnumerator PerformBlitzAttacks(EnemyAI target, int attackCount)
    {
        for (int i = 0; i < attackCount; i++)
        {
            yield return new WaitForSeconds(0.3f);
            target.TakeDamage(blitzDamage / attackCount);
        }
        
        // 원래 위치로 복귀
        yield return new WaitForSeconds(0.5f);
        // 원래 위치 복귀 로직은 별도로 구현 필요
    }
    
    private System.Collections.IEnumerator RemoveStealthAfterTime(float time)
    {
        yield return new WaitForSeconds(time);
        isInStealth = false;
        stats.evasion -= 30;
        TriggerDialogue("은신 해제", "은신이 풀렸어.");
    }
    
    private System.Collections.IEnumerator RemoveEvasiveAfterTime(float time)
    {
        yield return new WaitForSeconds(time);
        stats.evasion -= 50;
        TriggerDialogue("회피 해제", "회피 효과가 사라졌어.");
    }
    
    public override void TakeDamage(float damage, BodyPartType targetPart = BodyPartType.Torso)
    {
        // 제로는 은신 중일 때 피해를 70% 감소
        if (isInStealth)
        {
            damage *= 0.3f;
            TriggerDialogue("은신 중 피격", "아직도 날 찾을 수 있어?");
        }
        
        base.TakeDamage(damage, targetPart);
    }
    
    public void ScoutPath()
    {
        if (!CanUseSkill("ScoutPath") || stats.currentAP < 1) return;
        
        // 경로 정찰: 위험 지역을 피한 안전한 경로 제안
        // 실제 구현에서는 맵 시스템과 연동 필요
        UseSkill("ScoutPath", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("경로 정찰", "이쪽이 더 안전해 보여!");
    }
}

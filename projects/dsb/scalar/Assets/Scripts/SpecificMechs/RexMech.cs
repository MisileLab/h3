using UnityEngine;

public class RexMech : MechCharacter
{
    [Header("렉스 특수 능력")]
    public float guardianShieldStrength = 50f;
    public float tauntRange = 3f;
    public int tauntDuration = 2;
    
    private void Start()
    {
        mechType = MechType.Rex;
        mechName = "렉스";
        
        // 렉스 스탯 설정
        stats.maxHP = 150;
        stats.currentHP = 150;
        stats.maxAP = 3;
        stats.currentAP = 3;
        stats.speed = 8;
        stats.attack = 25;
        stats.defense = 30;
        stats.accuracy = 75;
        stats.evasion = 10;
        stats.leadership = 20;
        
        InitializeBodyParts();
        InitializeTrustLevels();
        InitializeCoopSkills();
    }
    
    public void GuardianShield(MechCharacter target)
    {
        if (!CanUseSkill("GuardianShield") || stats.currentAP < 2) return;
        
        // 가디언 실드 생성
        target.isGuarding = true;
        UseSkill("GuardianShield", 3f);
        ConsumeAP(2);
        
        TriggerDialogue("가디언 실드", "뒤로 물러나! 내가 막을게!");
        
        // 실드 효과를 일정 시간 후 제거
        StartCoroutine(RemoveShieldAfterTime(target, 2f));
    }
    
    public void TauntEnemies()
    {
        if (!CanUseSkill("Taunt") || stats.currentAP < 1) return;
        
        // 주변 적들을 도발하여 자신을 공격하도록 함
        Collider[] enemies = Physics.OverlapSphere(transform.position, tauntRange);
        
        foreach (Collider enemy in enemies)
        {
            EnemyAI enemyAI = enemy.GetComponent<EnemyAI>();
            if (enemyAI != null)
            {
                enemyAI.SetTauntedTarget(this, tauntDuration);
            }
        }
        
        UseSkill("Taunt", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("도발", "여기로 와! 나와 싸워!");
    }
    
    public void ProtectiveStance()
    {
        if (!CanUseSkill("ProtectiveStance") || stats.currentAP < 1) return;
        
        // 방어력 증가 및 아군 보호 모드
        stats.defense += 10;
        isGuarding = true;
        UseSkill("ProtectiveStance", 4f);
        ConsumeAP(1);
        
        TriggerDialogue("방어 태세", "아무도 내 동료들을 건드릴 수 없어!");
        
        // 3턴 후 효과 제거
        StartCoroutine(RemoveProtectiveStanceAfterTime(3f));
    }
    
    private System.Collections.IEnumerator RemoveShieldAfterTime(MechCharacter target, float time)
    {
        yield return new WaitForSeconds(time);
        target.isGuarding = false;
        TriggerDialogue("실드 해제", "실드가 사라졌어. 조심해!");
    }
    
    private System.Collections.IEnumerator RemoveProtectiveStanceAfterTime(float time)
    {
        yield return new WaitForSeconds(time);
        stats.defense -= 10;
        isGuarding = false;
        TriggerDialogue("방어 해제", "방어 태세를 해제했어.");
    }
    
    public override void TakeDamage(float damage, BodyPartType targetPart = BodyPartType.Torso)
    {
        // 렉스는 가드 중일 때 피해를 50% 감소
        if (isGuarding)
        {
            damage *= 0.5f;
            TriggerDialogue("가드", "이 정도는 괜찮아!");
        }
        
        base.TakeDamage(damage, targetPart);
    }
    
    public void GuardAlly(MechCharacter ally)
    {
        if (!CanUseSkill("GuardAlly") || stats.currentAP < 1) return;
        
        // 동료를 보호하는 행동
        ally.isGuarding = true;
        UseSkill("GuardAlly", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("동료 보호", $"{ally.mechName}! 내가 막을게!");
        IncreaseTrust(ally.mechType, 5);
    }
}

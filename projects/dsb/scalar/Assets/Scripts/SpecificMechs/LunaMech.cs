using UnityEngine;

public class LunaMech : MechCharacter
{
    [Header("루나 특수 능력")]
    public float repairAmount = 30f;
    public float hackSuccessRate = 0.8f;
    public int hackRange = 4;
    
    private void Start()
    {
        mechType = MechType.Luna;
        mechName = "루나";
        
        // 루나 스탯 설정
        stats.maxHP = 100;
        stats.currentHP = 100;
        stats.maxAP = 4;
        stats.currentAP = 4;
        stats.speed = 12;
        stats.attack = 15;
        stats.defense = 10;
        stats.accuracy = 90;
        stats.evasion = 25;
        stats.hackSkill = 25;
        stats.repairSkill = 30;
        
        InitializeBodyParts();
        InitializeTrustLevels();
        InitializeCoopSkills();
    }
    
    public void NanoRepair(MechCharacter target, BodyPartType partType)
    {
        if (!CanUseSkill("NanoRepair") || stats.currentAP < 2) return;
        
        // 나노 수리로 부위 복구
        target.RepairBodyPart(partType, repairAmount);
        UseSkill("NanoRepair", 3f);
        ConsumeAP(2);
        
        TriggerDialogue("나노 수리", "괜찮아, 내가 고쳐줄게!");
        IncreaseTrust(target.mechType, 8);
    }
    
    public void HackEnemy(EnemyAI enemy)
    {
        if (!CanUseSkill("Hack") || stats.currentAP < 2) return;
        
        float distance = Vector3.Distance(transform.position, enemy.transform.position);
        if (distance > hackRange) return;
        
        // 해킹 시도
        float successChance = hackSuccessRate + (stats.hackSkill / 100f);
        bool success = Random.Range(0f, 1f) < successChance;
        
        if (success)
        {
            enemy.GetHacked(2f); // 2초간 해킹 상태
            TriggerDialogue("해킹 성공", "적의 시스템을 침투했어!");
        }
        else
        {
            TriggerDialogue("해킹 실패", "방어 시스템이 강하네...");
        }
        
        UseSkill("Hack", 4f);
        ConsumeAP(2);
    }
    
    public void EmergencyRepair()
    {
        if (!CanUseSkill("EmergencyRepair") || stats.currentAP < 3) return;
        
        // 모든 아군의 가장 손상된 부위를 긴급 수리
        MechCharacter[] allies = FindObjectsOfType<MechCharacter>();
        
        foreach (MechCharacter ally in allies)
        {
            if (ally != this && ally.isAlive)
            {
                MechBodyPart mostDamaged = GetMostDamagedPart(ally);
                if (mostDamaged != null)
                {
                    ally.RepairBodyPart(mostDamaged.partType, repairAmount * 1.5f);
                }
            }
        }
        
        UseSkill("EmergencyRepair", 5f);
        ConsumeAP(3);
        
        TriggerDialogue("긴급 수리", "모두 괜찮아질 거야!");
    }
    
    public void SystemAnalysis(EnemyAI enemy)
    {
        if (!CanUseSkill("SystemAnalysis") || stats.currentAP < 1) return;
        
        // 적의 약점 분석
        enemy.RevealWeakness();
        UseSkill("SystemAnalysis", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("시스템 분석", "적의 약점을 찾았어!");
    }
    
    public void SupportBoost(MechCharacter target)
    {
        if (!CanUseSkill("SupportBoost") || stats.currentAP < 2) return;
        
        // 아군의 능력치 일시 상승
        target.stats.accuracy += 20;
        target.stats.evasion += 15;
        UseSkill("SupportBoost", 3f);
        ConsumeAP(2);
        
        TriggerDialogue("지원 강화", $"{target.mechName}, 더 잘할 수 있어!");
        IncreaseTrust(target.mechType, 5);
        
        // 3턴 후 효과 제거
        StartCoroutine(RemoveBoostAfterTime(target, 3f));
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
    
    private System.Collections.IEnumerator RemoveBoostAfterTime(MechCharacter target, float time)
    {
        yield return new WaitForSeconds(time);
        target.stats.accuracy -= 20;
        target.stats.evasion -= 15;
        TriggerDialogue("지원 해제", "지원 효과가 사라졌어.");
    }
    
    public void NegotiateWithEnemy(EnemyAI enemy)
    {
        if (!CanUseSkill("Negotiate") || stats.currentAP < 2) return;
        
        // 일부 지능형 적과 협상 시도
        if (enemy.CanBeNegotiated())
        {
            bool success = Random.Range(0f, 1f) < 0.6f;
            if (success)
            {
                enemy.ConvertToAlly();
                TriggerDialogue("협상 성공", "적이 우리 편이 되었어!");
            }
            else
            {
                TriggerDialogue("협상 실패", "아직 우리를 믿지 않네...");
            }
        }
        else
        {
            TriggerDialogue("협상 불가", "이 적과는 대화가 안 통해.");
        }
        
        UseSkill("Negotiate", 3f);
        ConsumeAP(2);
    }
}

using UnityEngine;

public class NovaMech : MechCharacter
{
    [Header("노바 특수 능력")]
    public float concentratedFireDamage = 80f;
    public float aoeRadius = 3f;
    public int aoeDamage = 50;
    
    private void Start()
    {
        mechType = MechType.Nova;
        mechName = "노바";
        
        // 노바 스탯 설정
        stats.maxHP = 120;
        stats.currentHP = 120;
        stats.maxAP = 3;
        stats.currentAP = 3;
        stats.speed = 6;
        stats.attack = 40;
        stats.defense = 20;
        stats.accuracy = 70;
        stats.evasion = 15;
        
        InitializeBodyParts();
        InitializeTrustLevels();
        InitializeCoopSkills();
    }
    
    public void ConcentratedFire(EnemyAI target)
    {
        if (!CanUseSkill("ConcentratedFire") || stats.currentAP < 2) return;
        
        // 집중 포격: 특정 적에게 강력한 단일 공격
        float damage = concentratedFireDamage + (stats.attack * 0.5f);
        target.TakeDamage(damage);
        
        UseSkill("ConcentratedFire", 3f);
        ConsumeAP(2);
        
        TriggerDialogue("집중 포격", "이것도 못 뚫으면 뭘로 뚫어!");
    }
    
    public void AreaBombardment(Vector3 targetPosition)
    {
        if (!CanUseSkill("AreaBombardment") || stats.currentAP < 3) return;
        
        // 지역 포격: 특정 지역에 강력한 범위 공격
        Collider[] targets = Physics.OverlapSphere(targetPosition, aoeRadius);
        
        foreach (Collider target in targets)
        {
            EnemyAI enemy = target.GetComponent<EnemyAI>();
            if (enemy != null)
            {
                enemy.TakeDamage(aoeDamage);
            }
            
            // 아군도 피격될 수 있음 (주의 필요)
            MechCharacter ally = target.GetComponent<MechCharacter>();
            if (ally != null && ally != this)
            {
                ally.TakeDamage(aoeDamage * 0.5f); // 아군에게는 절반 피해
                TriggerDialogue("아군 피격", "미안해! 조심해야겠어!");
            }
        }
        
        UseSkill("AreaBombardment", 4f);
        ConsumeAP(3);
        
        TriggerDialogue("지역 포격", "이 지역을 정리해주지!");
    }
    
    public void Overcharge()
    {
        if (!CanUseSkill("Overcharge") || stats.currentAP < 2) return;
        
        // 과충전: 다음 공격의 위력을 대폭 증가
        stats.attack += 20;
        UseSkill("Overcharge", 2f);
        ConsumeAP(2);
        
        TriggerDialogue("과충전", "최대 출력으로 가자!");
        
        // 2턴 후 효과 제거
        StartCoroutine(RemoveOverchargeAfterTime(2f));
    }
    
    public void SuppressiveFire(EnemyAI target)
    {
        if (!CanUseSkill("SuppressiveFire") || stats.currentAP < 1) return;
        
        // 억제 사격: 적의 행동을 제한
        target.ApplySuppression(2f);
        UseSkill("SuppressiveFire", 2f);
        ConsumeAP(1);
        
        TriggerDialogue("억제 사격", "움직이지 마!");
    }
    
    public void LinkAttack(MechCharacter ally, EnemyAI target)
    {
        if (!CanUseSkill("LinkAttack") || stats.currentAP < 2) return;
        
        // 연계 공격: 아군과 함께 공격
        float distance = Vector3.Distance(transform.position, ally.transform.position);
        if (distance <= 3f)
        {
            float combinedDamage = (stats.attack + ally.stats.attack) * 0.8f;
            target.TakeDamage(combinedDamage);
            
            UseSkill("LinkAttack", 3f);
            ConsumeAP(2);
            
            TriggerDialogue("연계 공격", $"{ally.mechName}과 함께!");
            IncreaseTrust(ally.mechType, 5);
        }
        else
        {
            TriggerDialogue("거리 부족", "너무 멀어서 연계할 수 없어.");
        }
    }
    
    public void ArtilleryStrike(Vector3 targetPosition)
    {
        if (!CanUseSkill("ArtilleryStrike") || stats.currentAP < 3) return;
        
        // 포격: 원거리에서 강력한 공격
        StartCoroutine(PerformArtilleryStrike(targetPosition));
        
        UseSkill("ArtilleryStrike", 5f);
        ConsumeAP(3);
        
        TriggerDialogue("포격", "원거리에서 지원할게!");
    }
    
    public void DefensiveBarrage()
    {
        if (!CanUseSkill("DefensiveBarrage") || stats.currentAP < 2) return;
        
        // 방어 포격: 주변 적들을 견제
        Collider[] enemies = Physics.OverlapSphere(transform.position, 5f);
        
        foreach (Collider enemy in enemies)
        {
            EnemyAI enemyAI = enemy.GetComponent<EnemyAI>();
            if (enemyAI != null)
            {
                enemyAI.TakeDamage(stats.attack * 0.6f);
                enemyAI.ApplySuppression(1f);
            }
        }
        
        UseSkill("DefensiveBarrage", 3f);
        ConsumeAP(2);
        
        TriggerDialogue("방어 포격", "가까이 오지 마!");
    }
    
    private System.Collections.IEnumerator RemoveOverchargeAfterTime(float time)
    {
        yield return new WaitForSeconds(time);
        stats.attack -= 20;
        TriggerDialogue("과충전 해제", "출력이 정상으로 돌아왔어.");
    }
    
    private System.Collections.IEnumerator PerformArtilleryStrike(Vector3 targetPosition)
    {
        // 2초 후 포격 실행
        yield return new WaitForSeconds(2f);
        
        Collider[] targets = Physics.OverlapSphere(targetPosition, aoeRadius * 1.5f);
        
        foreach (Collider target in targets)
        {
            EnemyAI enemy = target.GetComponent<EnemyAI>();
            if (enemy != null)
            {
                enemy.TakeDamage(aoeDamage * 1.5f);
            }
        }
        
        TriggerDialogue("포격 완료", "목표 타격 완료!");
    }
    
    public void HeavyWeaponMode()
    {
        if (!CanUseSkill("HeavyWeaponMode") || stats.currentAP < 2) return;
        
        // 중화기 모드: 공격력 증가하지만 이동 불가
        stats.attack += 30;
        stats.speed = 0; // 이동 불가
        UseSkill("HeavyWeaponMode", 4f);
        ConsumeAP(2);
        
        TriggerDialogue("중화기 모드", "이제 진짜 시작이야!");
        
        // 3턴 후 효과 제거
        StartCoroutine(RemoveHeavyWeaponModeAfterTime(3f));
    }
    
    private System.Collections.IEnumerator RemoveHeavyWeaponModeAfterTime(float time)
    {
        yield return new WaitForSeconds(time);
        stats.attack -= 30;
        stats.speed = 6; // 원래 속도로 복구
        TriggerDialogue("중화기 해제", "다시 움직일 수 있어.");
    }
    
    public override void TakeDamage(float damage, BodyPartType targetPart = BodyPartType.Torso)
    {
        // 노바는 중화기 모드일 때 피해를 25% 감소
        if (stats.speed == 0)
        {
            damage *= 0.75f;
            TriggerDialogue("중화기 방어", "이 정도는 버틸 수 있어!");
        }
        
        base.TakeDamage(damage, targetPart);
    }
}

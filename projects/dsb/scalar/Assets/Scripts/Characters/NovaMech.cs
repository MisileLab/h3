using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

/// <summary>
/// 노바 (Nova) - 헤비 어태커 (광역 딜러)
/// 핵심 능력: 집중 포격 - 특정 지역에 강력한 범위 공격을 가함 (아군 피격 주의)
/// 전투 철학: "이것도 못 뚫으면 뭘로 뚫어!"
/// </summary>
public class NovaMech : MechCharacter
{
    [Header("노바 전용 설정")]
    public int artilleryDamage = 45;                // 집중 포격 피해량
    public float artilleryRadius = 2.5f;            // 집중 포격 반경
    public float artilleryRange = 8f;               // 집중 포격 사거리
    public float artilleryCooldown = 0f;            // 집중 포격 쿨다운
    public float artilleryMaxCooldown = 4f;         // 집중 포격 최대 쿨다운
    public bool artilleryFriendlyFire = true;       // 아군 피격 가능 여부
    
    [Header("중화기 시스템")]
    public int heavyWeaponAmmo = 3;                 // 중화기 탄약
    public int maxHeavyWeaponAmmo = 3;              // 최대 탄약
    public float reloadCooldown = 0f;               // 재장전 쿨다운
    public float reloadMaxCooldown = 2f;            // 재장전 최대 쿨다운
    public int heavyWeaponDamage = 35;              // 중화기 피해량
    public float heavyWeaponRange = 6f;             // 중화기 사거리
    
    [Header("방어 시스템")]
    public bool isInSiegeMode = false;              // 공성 모드 여부
    public int siegeModeDefenseBonus = 20;          // 공성 모드 방어력 보너스
    public float siegeModeDamageBonus = 1.5f;       // 공성 모드 피해 보너스
    public int siegeModeMovementPenalty = -2;       // 공성 모드 이동 패널티
    
    [Header("과부하 시스템")]
    public bool isOverloaded = false;               // 과부하 상태
    public int overloadTurns = 0;                   // 과부하 지속 턴
    public float overloadDamageMultiplier = 2f;     // 과부하 피해 배율
    public int overloadSelfDamage = 15;             // 과부하 자가 피해
    
    [Header("지원 공격")]
    public List<Vector3> bombardmentTargets;        // 폭격 타겟 리스트
    
    protected override void Start()
    {
        // 기본 정보 설정
        mechName = "노바";
        mechType = MechType.Nova;
        
        // 노바의 스탯 (화력 특화)
        stats = new MechStats
        {
            maxHP = 110,        // 높은 HP (중장갑)
            currentHP = 110,
            attack = 30,        // 최고 공격력
            defense = 25,       // 높은 방어력
            speed = 4,          // 최저 속도
            accuracy = 70,      // 낮은 명중률 (중화기의 단점)
            evasion = 5         // 최저 회피율
        };
        
        bombardmentTargets = new List<Vector3>();
        
        base.Start();
    }
    
    protected override void InitializeDialogues()
    {
        dialogueLines = new List<DialogueLine>
        {
            // 턴 시작
            new DialogueLine { type = DialogueType.TurnStart, text = "이것도 못 뚫으면 뭘로 뚫어!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "화력 지원 준비 완료!", weight = 1f },
            new DialogueLine { type = DialogueType.TurnStart, text = "중화기 시스템 온라인!", weight = 0.8f },
            
            // 피해 받을 때
            new DialogueLine { type = DialogueType.Damage, text = "장갑이 단단해서 다행이야!", weight = 1f },
            new DialogueLine { type = DialogueType.Damage, text = "이 정도로 날 멈출 순 없어!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "주포 시스템이 손상됐어!", weight = 1f },
            new DialogueLine { type = DialogueType.SevereDamage, text = "화력이... 떨어지고 있어!", weight = 1f },
            
            // 치료 받을 때
            new DialogueLine { type = DialogueType.Healed, text = "좋아! 다시 풀 파워다!", weight = 1f },
            new DialogueLine { type = DialogueType.Healed, text = "시스템 복구 완료. 계속 쏠 수 있어!", weight = 1f },
            
            // 집중 포격
            new DialogueLine { type = DialogueType.Cooperative, text = "집중 포격 준비! 모두 엄폐해!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "화력 집중! 적을 날려버리자!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "폭격 지점 마킹 완료! 발사!", weight = 1f },
            
            // 중화기 공격
            new DialogueLine { type = DialogueType.Cooperative, text = "중화기 발사!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "이 한 방이면 충분해!", weight = 1f },
            
            // 탄약 부족
            new DialogueLine { type = DialogueType.Damage, text = "탄약이 부족해! 재장전 필요!", weight = 1f },
            new DialogueLine { type = DialogueType.Worried, text = "잠깐, 재장전할 시간을 줘!", weight = 1f },
            
            // 과부하 상태
            new DialogueLine { type = DialogueType.Critical, text = "시스템 과부하! 위험 수치야!", weight = 1f },
            new DialogueLine { type = DialogueType.Critical, text = "엔진이 과열되고 있어!", weight = 1f },
            
            // 공성 모드
            new DialogueLine { type = DialogueType.Cooperative, text = "공성 모드 전환! 포지션 고정!", weight = 1f },
            new DialogueLine { type = DialogueType.Cooperative, text = "이제 움직일 수 없지만, 더 강해졌어!", weight = 1f },
            
            // 부위 파괴 (특히 오른팔 - 주무장)
            new DialogueLine { type = DialogueType.PartDestroyed, text = "주포가...! 보조 화기로 전환!", weight = 1f },
            
            // 전투 불능
            new DialogueLine { type = DialogueType.Incapacitated, text = "탄약... 고갈... 미안해...", weight = 1f },
            new DialogueLine { type = DialogueType.Incapacitated, text = "더 이상... 지원할 수 없어...", weight = 1f },
            
            // 회복
            new DialogueLine { type = DialogueType.Revived, text = "화력 시스템 재가동! 다시 싸울 수 있어!", weight = 1f },
            new DialogueLine { type = DialogueType.Revived, text = "탄약 보급 완료! 전투 속행!", weight = 1f },
            
            // 승리
            new DialogueLine { type = DialogueType.Victory, text = "화력이 모든 걸 해결했어!", weight = 1f },
            new DialogueLine { type = DialogueType.Victory, text = "완벽한 화력 지원이었어!", weight = 1f }
        };
    }
    
    public override void StartTurn()
    {
        base.StartTurn();
        
        // 과부하 지속 시간 감소
        if (isOverloaded && overloadTurns > 0)
        {
            overloadTurns--;
            
            // 과부하 자가 피해
            TakeDamage(overloadSelfDamage);
            SayDialogue("시스템 과부하로 인한 내부 손상!", DialogueType.Damage);
            
            if (overloadTurns <= 0)
            {
                ExitOverload();
            }
        }
    }
    
    public override void UpdateCooldowns(float deltaTime)
    {
        base.UpdateCooldowns(deltaTime);
        
        // 집중 포격 쿨다운 업데이트
        if (artilleryCooldown > 0)
        {
            artilleryCooldown -= deltaTime;
            if (artilleryCooldown <= 0)
            {
                artilleryCooldown = 0;
                Debug.Log($"{mechName}의 집중 포격 쿨다운이 완료되었습니다.");
            }
        }
        
        // 재장전 쿨다운 업데이트
        if (reloadCooldown > 0)
        {
            reloadCooldown -= deltaTime;
            if (reloadCooldown <= 0)
            {
                reloadCooldown = 0;
                Debug.Log($"{mechName}의 재장전이 완료되었습니다.");
            }
        }
    }
    
    /// <summary>
    /// 집중 포격 - 특정 지역에 강력한 범위 공격 (아군 피격 주의)
    /// </summary>
    /// <param name="targetPosition">포격 위치</param>
    /// <param name="warningTurns">경고 턴 (아군이 피할 시간)</param>
    /// <returns>성공하면 true</returns>
    public bool UseConcentratedBombardment(Vector3 targetPosition, int warningTurns = 1)
    {
        // AP 및 쿨다운 확인
        if (!actionPoints.CanUseAP(3))
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 3)");
            return false;
        }
        
        if (artilleryCooldown > 0)
        {
            Debug.LogWarning($"집중 포격이 쿨다운 중입니다. ({artilleryCooldown:F1}초 남음)");
            return false;
        }
        
        // 사거리 확인
        float distance = Vector3.Distance(transform.position, targetPosition);
        if (distance > artilleryRange)
        {
            Debug.LogWarning($"포격 지점이 너무 멀어요. (거리: {distance:F1}, 최대: {artilleryRange})");
            return false;
        }
        
        // AP 소모
        actionPoints.ConsumeAP(3);
        
        // 포격 실행
        StartCoroutine(BombardmentSequence(targetPosition, warningTurns));
        
        // 쿨다운 설정
        artilleryCooldown = artilleryMaxCooldown;
        
        return true;
    }
    
    /// <summary>
    /// 집중 포격 시퀀스 코루틴
    /// </summary>
    /// <param name="targetPosition">포격 위치</param>
    /// <param name="warningTurns">경고 턴</param>
    private IEnumerator BombardmentSequence(Vector3 targetPosition, int warningTurns)
    {
        // 경고 발사
        SayDialogue("집중 포격 준비! 모두 엄폐해!", DialogueType.Cooperative);
        
        // 경고 시간 (아군이 피할 수 있도록)
        for (int i = 0; i < warningTurns; i++)
        {
            Debug.Log($"<color=orange>집중 포격 경고! {warningTurns - i}턴 후 발사!</color>");
            
            // 아군들에게 경고 대사
            var allies = FindObjectsOfType<MechCharacter>().Where(m => m != this && m.isAlive).ToList();
            foreach (var ally in allies)
            {
                float distance = Vector3.Distance(ally.transform.position, targetPosition);
                if (distance <= artilleryRadius)
                {
                    ally.SayDialogue($"{mechName}! 포격 지점에서 벗어나야 해!", DialogueType.Worried);
                    break;
                }
            }
            
            yield return new WaitForSeconds(1f);
        }
        
        // 실제 포격 실행
        ExecuteBombardment(targetPosition);
    }
    
    /// <summary>
    /// 실제 포격 실행
    /// </summary>
    /// <param name="targetPosition">포격 위치</param>
    private void ExecuteBombardment(Vector3 targetPosition)
    {
        SayDialogue("집중 포격 발사!", DialogueType.Cooperative);
        
        // 피해 계산
        int baseDamage = artilleryDamage;
        if (isInSiegeMode)
        {
            baseDamage = Mathf.RoundToInt(baseDamage * siegeModeDamageBonus);
        }
        if (isOverloaded)
        {
            baseDamage = Mathf.RoundToInt(baseDamage * overloadDamageMultiplier);
        }
        
        // 범위 내 모든 유닛에게 피해 적용
        var allTargets = new List<MonoBehaviour>();
        allTargets.AddRange(FindObjectsOfType<EnemyAI>());
        
        if (artilleryFriendlyFire)
        {
            allTargets.AddRange(FindObjectsOfType<MechCharacter>());
        }
        
        int hitCount = 0;
        int friendlyFireCount = 0;
        
        foreach (var target in allTargets)
        {
            float distance = Vector3.Distance(target.transform.position, targetPosition);
            if (distance <= artilleryRadius)
            {
                // 거리에 따른 피해 감소
                float damageMultiplier = 1f - (distance / artilleryRadius) * 0.5f; // 최대 50% 감소
                int actualDamage = Mathf.RoundToInt(baseDamage * damageMultiplier);
                
                if (target is EnemyAI enemy)
                {
                    enemy.TakeDamage(actualDamage);
                    hitCount++;
                }
                else if (target is MechCharacter ally && ally != this)
                {
                    ally.TakeDamage(actualDamage);
                    friendlyFireCount++;
                    
                    // 아군 피격 시 사과 대사
                    if (UnityEngine.Random.Range(0f, 1f) < 0.5f)
                    {
                        SayDialogue($"미안해, {ally.mechName}! 너무 가까이 있었어!", DialogueType.Worried);
                        ally.SayDialogue($"{mechName}, 조준을 좀 더 정확히 해줘!", DialogueType.Damage);
                    }
                }
            }
        }
        
        Debug.Log($"<color=red>집중 포격 완료! 적 {hitCount}마리 피격" + 
                 (friendlyFireCount > 0 ? $", 아군 {friendlyFireCount}명 오사" : "") + "</color>");
    }
    
    /// <summary>
    /// 중화기 공격 - 단발 고위력 공격
    /// </summary>
    /// <param name="target">공격 대상</param>
    /// <returns>성공하면 true</returns>
    public bool UseHeavyWeapon(EnemyAI target)
    {
        // AP 및 탄약 확인
        if (!actionPoints.CanUseAP(2))
        {
            Debug.LogWarning("AP가 부족합니다. (필요: 2)");
            return false;
        }
        
        if (heavyWeaponAmmo <= 0)
        {
            SayDialogue("탄약이 부족해! 재장전 필요!", DialogueType.Damage);
            Debug.LogWarning("중화기 탄약이 없습니다.");
            return false;
        }
        
        // 사거리 확인
        float distance = Vector3.Distance(transform.position, target.transform.position);
        if (distance > heavyWeaponRange)
        {
            Debug.LogWarning($"대상이 너무 멀리 있습니다. (거리: {distance:F1}, 최대: {heavyWeaponRange})");
            return false;
        }
        
        // AP 및 탄약 소모
        actionPoints.ConsumeAP(2);
        heavyWeaponAmmo--;
        
        // 피해 계산
        int damage = heavyWeaponDamage;
        if (isInSiegeMode)
        {
            damage = Mathf.RoundToInt(damage * siegeModeDamageBonus);
        }
        if (isOverloaded)
        {
            damage = Mathf.RoundToInt(damage * overloadDamageMultiplier);
        }
        
        // 공격 실행
        target.TakeDamage(damage);
        
        SayDialogue("중화기 발사!", DialogueType.Cooperative);
        
        Debug.Log($"<color=red>{mechName}이 {target.enemyName}에게 중화기로 {damage} 피해!</color>");
        Debug.Log($"남은 중화기 탄약: {heavyWeaponAmmo}/{maxHeavyWeaponAmmo}");
        
        // 탄약이 떨어지면 알림
        if (heavyWeaponAmmo == 0)
        {
            SayDialogue("탄약 고갈! 재장전이 필요해!", DialogueType.Worried);
        }
        
        return true;
    }
    
    /// <summary>
    /// 재장전 - 중화기 탄약을 보충
    /// </summary>
    /// <returns>성공하면 true</returns>
    public bool UseReload()
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        if (reloadCooldown > 0)
        {
            Debug.LogWarning($"재장전이 쿨다운 중입니다. ({reloadCooldown:F1}초 남음)");
            return false;
        }
        
        if (heavyWeaponAmmo >= maxHeavyWeaponAmmo)
        {
            Debug.LogWarning("이미 탄약이 가득합니다.");
            return false;
        }
        
        actionPoints.ConsumeAP(2);
        heavyWeaponAmmo = maxHeavyWeaponAmmo;
        reloadCooldown = reloadMaxCooldown;
        
        SayDialogue("재장전 완료! 다시 쏠 수 있어!", DialogueType.Cooperative);
        
        Debug.Log($"<color=yellow>{mechName}이 재장전했습니다. 탄약: {heavyWeaponAmmo}/{maxHeavyWeaponAmmo}</color>");
        return true;
    }
    
    /// <summary>
    /// 공성 모드 전환 - 이동 불가, 방어력 및 공격력 증가
    /// </summary>
    /// <returns>성공하면 true</returns>
    public bool ToggleSiegeMode()
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        actionPoints.ConsumeAP(1);
        
        isInSiegeMode = !isInSiegeMode;
        
        if (isInSiegeMode)
        {
            SayDialogue("공성 모드 전환! 포지션 고정!", DialogueType.Cooperative);
            Debug.Log($"<color=orange>{mechName}이 공성 모드로 전환했습니다. 방어력 +{siegeModeDefenseBonus}, 공격력 +{(int)((siegeModeDamageBonus - 1) * 100)}%</color>");
        }
        else
        {
            SayDialogue("공성 모드 해제! 기동성 복구!", DialogueType.Cooperative);
            Debug.Log($"<color=orange>{mechName}이 공성 모드를 해제했습니다.</color>");
        }
        
        return true;
    }
    
    /// <summary>
    /// 과부하 모드 - 강력한 공격력, 하지만 자가 피해
    /// </summary>
    /// <param name="duration">지속 턴</param>
    /// <returns>성공하면 true</returns>
    public bool UseOverload(int duration = 2)
    {
        if (!actionPoints.CanUseAP(2)) return false;
        
        if (isOverloaded)
        {
            Debug.LogWarning("이미 과부하 상태입니다.");
            return false;
        }
        
        actionPoints.ConsumeAP(2);
        
        isOverloaded = true;
        overloadTurns = duration;
        
        SayDialogue("시스템 과부하 모드! 최대 화력!", DialogueType.Critical);
        
        Debug.Log($"<color=red>{mechName}이 과부하 모드로 전환했습니다! 공격력 +{(int)((overloadDamageMultiplier - 1) * 100)}%, {duration}턴 지속</color>");
        return true;
    }
    
    /// <summary>
    /// 과부하 모드 해제
    /// </summary>
    private void ExitOverload()
    {
        isOverloaded = false;
        overloadTurns = 0;
        
        SayDialogue("시스템 과부하 해제... 정상 출력으로 복귀.", DialogueType.Cooperative);
        Debug.Log($"<color=orange>{mechName}의 과부하 모드가 해제되었습니다.</color>");
    }
    
    /// <summary>
    /// 지원 사격 지점 설정
    /// </summary>
    /// <param name="position">지원 사격 위치</param>
    /// <returns>성공하면 true</returns>
    public bool SetSupportFirePosition(Vector3 position)
    {
        if (!actionPoints.CanUseAP(1)) return false;
        
        actionPoints.ConsumeAP(1);
        
        if (!bombardmentTargets.Contains(position))
        {
            bombardmentTargets.Add(position);
        }
        
        SayDialogue("지원 사격 지점 마킹 완료!", DialogueType.Cooperative);
        
        Debug.Log($"<color=yellow>{mechName}이 지원 사격 지점을 설정했습니다. 총 {bombardmentTargets.Count}개 지점</color>");
        return true;
    }
    
    /// <summary>
    /// 모든 마킹된 지점에 지원 사격 실행
    /// </summary>
    /// <returns>타격한 지점 수</returns>
    public int ExecuteSupportFire()
    {
        if (!actionPoints.CanUseAP(3)) return 0;
        
        if (bombardmentTargets.Count == 0)
        {
            Debug.LogWarning("지원 사격 지점이 설정되지 않았습니다.");
            return 0;
        }
        
        actionPoints.ConsumeAP(3);
        
        int hitTargets = 0;
        foreach (var position in bombardmentTargets)
        {
            ExecuteBombardment(position);
            hitTargets++;
        }
        
        bombardmentTargets.Clear();
        
        SayDialogue($"{hitTargets}개 지점에 지원 사격 완료!", DialogueType.Cooperative);
        
        Debug.Log($"<color=red>{mechName}이 {hitTargets}개 지점에 지원 사격을 완료했습니다!</color>");
        return hitTargets;
    }
    
    /// <summary>
    /// 노바는 높은 방어력으로 피해를 감소시킵니다
    /// </summary>
    public override int TakeDamage(int damage, BodyPartType? targetPart = null)
    {
        // 공성 모드 중 방어력 보너스
        if (isInSiegeMode)
        {
            damage = Mathf.RoundToInt(damage * (1f - siegeModeDefenseBonus / 100f));
            Debug.Log($"<color=orange>공성 모드 방어 보너스! 피해 {siegeModeDefenseBonus}% 감소</color>");
        }
        
        return base.TakeDamage(damage, targetPart);
    }
    
    /// <summary>
    /// 상태 정보를 문자열로 반환
    /// </summary>
    public override string ToString()
    {
        string status = $"{mechName} (HP: {stats.currentHP}/{stats.maxHP}, AP: {actionPoints.currentAP}/{actionPoints.maxAP})";
        
        status += $"\n중화기 탄약: {heavyWeaponAmmo}/{maxHeavyWeaponAmmo}";
        
        if (isInSiegeMode)
        {
            status += "\n공성 모드 활성화";
        }
        
        if (isOverloaded)
        {
            status += $"\n과부하 모드 ({overloadTurns}턴 남음)";
        }
        
        if (artilleryCooldown > 0)
        {
            status += $"\n집중 포격 쿨다운: {artilleryCooldown:F1}초";
        }
        
        if (reloadCooldown > 0)
        {
            status += $"\n재장전 쿨다운: {reloadCooldown:F1}초";
        }
        
        if (bombardmentTargets.Count > 0)
        {
            status += $"\n지원 사격 지점: {bombardmentTargets.Count}개";
        }
        
        return status;
    }
}

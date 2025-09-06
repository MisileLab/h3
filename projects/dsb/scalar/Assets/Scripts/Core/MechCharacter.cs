using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System;

/// <summary>
/// 기계 캐릭터의 기본 클래스
/// 모든 플레이어블 기계들의 공통 기능을 담당합니다.
/// </summary>
public abstract class MechCharacter : MonoBehaviour
{
    [Header("기본 정보")]
    public string mechName;
    public MechType mechType;
    public bool isAlive = true;
    
    [Header("스탯")]
    public MechStats stats;
    
    [Header("시스템")]
    public ActionPoint actionPoints;
    public List<BodyPart> bodyParts;
    public Dictionary<MechCharacter, int> trustLevels; // 신뢰도 시스템
    
    [Header("상태")]
    public MechCharacter currentProtector;    // 현재 나를 보호하는 기계
    public int protectionTurnsLeft = 0;       // 보호 지속 턴
    public bool isIncapacitated = false;      // 전투 불능 상태
    public bool isGuarding = false;           // 방어 태세 여부
    public bool isInStealth = false;          // 은신 여부
    public bool isInCombat = false;            // 전투 중 여부
    
    [Header("대화 시스템")]
    public List<DialogueLine> dialogueLines; // 상황별 대사들
    
    // 이벤트
    public static event Action<MechCharacter, int> OnHealthChanged;
    public static event Action<MechCharacter, BodyPart> OnBodyPartDestroyed;
    public static event Action<MechCharacter, string, DialogueType> OnDialogueTriggered;
    public static event Action<MechCharacter> OnMechIncapacitated;
    public static event Action<MechCharacter> OnMechRevived;
    
    protected virtual void Start()
    {
        InitializeMech();
    }
    
    protected virtual void Update()
    {
        UpdateProtectionStatus();
    }
    
    /// <summary>
    /// 기계 초기화
    /// </summary>
    protected virtual void InitializeMech()
    {
        // AP 시스템 초기화
        actionPoints = new ActionPoint(3); // 기본 3 AP
        
        // 부위별 HP 초기화
        InitializeBodyParts();
        
        // 신뢰도 시스템 초기화
        trustLevels = new Dictionary<MechCharacter, int>();
        
        // 대사 시스템 초기화
        InitializeDialogues();
        
        // 이벤트 구독
        BodyPart.OnPartDestroyed += OnBodyPartDestroyed_Handler;
        
        Debug.Log($"{mechName} 초기화 완료");
    }
    
    /// <summary>
    /// 부위별 HP 시스템 초기화
    /// </summary>
    protected virtual void InitializeBodyParts()
    {
        bodyParts = new List<BodyPart>
        {
            new BodyPart(BodyPartType.Head, 0.15f, stats.maxHP),      // 머리/센서 (15%)
            new BodyPart(BodyPartType.Torso, 0.40f, stats.maxHP),     // 몸통/코어 (40%)
            new BodyPart(BodyPartType.RightArm, 0.20f, stats.maxHP),  // 오른팔/주무장 (20%)
            new BodyPart(BodyPartType.LeftArm, 0.15f, stats.maxHP),   // 왼팔/보조장비 (15%)
            new BodyPart(BodyPartType.Legs, 0.10f, stats.maxHP)       // 다리/이동부 (10%)
        };
    }
    
    /// <summary>
    /// 대사 시스템 초기화 (하위 클래스에서 구현)
    /// </summary>
    protected abstract void InitializeDialogues();
    
    /// <summary>
    /// 턴 시작 처리
    /// </summary>
    public virtual void StartTurn()
    {
        // AP 복구
        actionPoints.RefreshAP();
        
        // 상태 업데이트
        UpdateStatuses();
        
        // 턴 시작 대사
        if (UnityEngine.Random.Range(0f, 1f) < 0.3f) // 30% 확률
        {
            SayDialogue(GetRandomDialogue(DialogueType.TurnStart), DialogueType.TurnStart);
        }
        
        Debug.Log($"{mechName}의 턴이 시작되었습니다. (AP: {actionPoints.currentAP})");
    }
    
    /// <summary>
    /// 턴 종료 처리
    /// </summary>
    public virtual void EndTurn()
    {
        // 보호 지속 시간 감소
        if (protectionTurnsLeft > 0)
        {
            protectionTurnsLeft--;
            if (protectionTurnsLeft <= 0)
            {
                currentProtector = null;
                Debug.Log($"{mechName}의 보호가 해제되었습니다.");
            }
        }
        
        Debug.Log($"{mechName}의 턴이 종료되었습니다.");
    }
    
    /// <summary>
    /// 피해를 받습니다
    /// </summary>
    /// <param name="damage">피해량</param>
    /// <param name="targetPart">특정 부위를 노린 공격 (null이면 랜덤)</param>
    /// <returns>실제로 입은 피해량</returns>
    public virtual int TakeDamage(int damage, BodyPartType? targetPart = null)
    {
        if (!isAlive) return 0;
        
        // 보호자가 있으면 보호자가 대신 받음
        if (currentProtector != null && currentProtector.isAlive)
        {
            Debug.Log($"<color=yellow>{currentProtector.mechName}이 {mechName}을 보호합니다!</color>");
            currentProtector.SayDialogue($"{mechName}! 내가 막을게!", DialogueType.Protective);
            return currentProtector.TakeDamage(damage, targetPart);
        }
        
        // 타겟 부위 결정
        BodyPart targetBodyPart;
        if (targetPart.HasValue)
        {
            targetBodyPart = bodyParts.FirstOrDefault(bp => bp.partType == targetPart.Value);
        }
        else
        {
            // 랜덤하게 부위 선택 (HP 비율에 따른 가중치)
            targetBodyPart = SelectRandomBodyPart();
        }
        
        if (targetBodyPart == null || targetBodyPart.isDestroyed)
        {
            Debug.LogWarning("유효한 타겟 부위를 찾을 수 없습니다.");
            return 0;
        }
        
        // 피해 적용
        int actualDamage = targetBodyPart.TakeDamage(damage);
        stats.currentHP = bodyParts.Where(bp => !bp.isDestroyed).Sum(bp => bp.currentHP);
        
        // 피해 받을 때 대사
        if (actualDamage > 0)
        {
            float damageRatio = (float)actualDamage / stats.maxHP;
            if (damageRatio > 0.2f) // 심각한 피해
            {
                SayDialogue(GetRandomDialogue(DialogueType.SevereDamage), DialogueType.SevereDamage);
            }
            else if (damageRatio > 0.1f) // 중간 피해
            {
                SayDialogue(GetRandomDialogue(DialogueType.Damage), DialogueType.Damage);
            }
        }
        
        // 전투 불능 확인
        CheckIncapacitation();
        
        OnHealthChanged?.Invoke(this, actualDamage);
        Debug.Log($"{mechName}이 {targetBodyPart.partName}에 {actualDamage} 피해를 받았습니다. (남은 HP: {stats.currentHP})");
        
        return actualDamage;
    }
    
    /// <summary>
    /// 치료를 받습니다
    /// </summary>
    /// <param name="healAmount">치료량</param>
    /// <param name="targetPart">치료할 특정 부위</param>
    /// <returns>실제 치료량</returns>
    public virtual int Heal(int healAmount, BodyPartType? targetPart = null)
    {
        if (!isAlive) return 0;
        
        BodyPart healTarget;
        if (targetPart.HasValue)
        {
            healTarget = bodyParts.FirstOrDefault(bp => bp.partType == targetPart.Value);
        }
        else
        {
            healTarget = GetMostDamagedPart();
        }
        
        if (healTarget == null) return 0;
        
        int actualHeal = healTarget.Heal(healAmount);
        stats.currentHP = bodyParts.Where(bp => !bp.isDestroyed).Sum(bp => bp.currentHP);
        
        // 치료 시 대사
        if (actualHeal > 0)
        {
            SayDialogue(GetRandomDialogue(DialogueType.Healed), DialogueType.Healed);
        }
        
        // 전투 불능 해제 확인
        if (isIncapacitated && CanRecoverFromIncapacitation())
        {
            RecoverFromIncapacitation();
        }
        
        OnHealthChanged?.Invoke(this, -actualHeal); // 음수로 치료량 표시
        Debug.Log($"{mechName}이 {healTarget.partName}을 {actualHeal} 치료받았습니다.");
        
        return actualHeal;
    }
    
    /// <summary>
    /// 보호자를 설정합니다
    /// </summary>
    /// <param name="protector">보호하는 기계</param>
    /// <param name="turns">지속 턴 수</param>
    public void SetProtector(MechCharacter protector, int turns)
    {
        currentProtector = protector;
        protectionTurnsLeft = turns;
        Debug.Log($"{protector.mechName}이 {mechName}을 보호합니다. ({turns}턴)");
    }
    
    /// <summary>
    /// 동료들과의 신뢰도를 증가시킵니다
    /// </summary>
    /// <param name="allies">신뢰도가 증가할 동료들</param>
    /// <param name="amount">증가량</param>
    public void IncreaseTrustWithAllies(List<MechCharacter> allies, int amount)
    {
        foreach (var ally in allies)
        {
            if (ally == this) continue;
            
            if (!trustLevels.ContainsKey(ally))
                trustLevels[ally] = 0;
            
            trustLevels[ally] += amount;
            Debug.Log($"{mechName}의 {ally.mechName}에 대한 신뢰도가 {amount} 증가했습니다. (현재: {trustLevels[ally]})");
        }
    }
    
    /// <summary>
    /// 대사를 출력합니다
    /// </summary>
    /// <param name="dialogue">대사 내용</param>
    /// <param name="type">대사 타입</param>
    public void SayDialogue(string dialogue, DialogueType type)
    {
        if (string.IsNullOrEmpty(dialogue)) return;
        
        OnDialogueTriggered?.Invoke(this, dialogue, type);
        Debug.Log($"<color=cyan>[{mechName}]</color> {dialogue}");
    }
    
    /// <summary>
    /// 특정 타입의 랜덤 대사를 반환합니다
    /// </summary>
    /// <param name="type">대사 타입</param>
    /// <returns>대사 문자열</returns>
    public string GetRandomDialogue(DialogueType type)
    {
        var availableDialogues = dialogueLines.Where(d => d.type == type).ToList();
        if (availableDialogues.Count == 0) return "";
        
        return availableDialogues[UnityEngine.Random.Range(0, availableDialogues.Count)].text;
    }
    
    /// <summary>
    /// 가장 손상이 심한 부위를 반환합니다
    /// </summary>
    /// <returns>가장 손상이 심한 부위</returns>
    public BodyPart GetMostDamagedPart()
    {
        return bodyParts.Where(bp => !bp.isDestroyed).OrderBy(bp => bp.GetHPRatio()).FirstOrDefault();
    }
    
    /// <summary>
    /// 랜덤하게 부위를 선택합니다 (HP 비율에 따른 가중치)
    /// </summary>
    /// <returns>선택된 부위</returns>
    protected BodyPart SelectRandomBodyPart()
    {
        var availableParts = bodyParts.Where(bp => !bp.isDestroyed).ToList();
        if (availableParts.Count == 0) return null;
        
        // HP 비율에 따른 가중치 계산 (HP가 높을수록 타겟이 될 확률이 높음)
        float totalWeight = availableParts.Sum(bp => bp.hpPercentage);
        float randomValue = UnityEngine.Random.Range(0f, totalWeight);
        
        float currentWeight = 0f;
        foreach (var part in availableParts)
        {
            currentWeight += part.hpPercentage;
            if (randomValue <= currentWeight)
                return part;
        }
        
        return availableParts.Last();
    }
    
    /// <summary>
    /// 전투 불능 상태를 확인합니다
    /// </summary>
    protected virtual void CheckIncapacitation()
    {
        // 몸통(코어)이 파괴되면 즉시 전투 불능
        var torso = bodyParts.FirstOrDefault(bp => bp.partType == BodyPartType.Torso);
        if (torso != null && torso.isDestroyed)
        {
            SetIncapacitated();
            return;
        }
        
        // 전체 HP가 0 이하면 전투 불능
        if (stats.currentHP <= 0)
        {
            SetIncapacitated();
            return;
        }
    }
    
    /// <summary>
    /// 전투 불능 상태로 설정합니다
    /// </summary>
    protected virtual void SetIncapacitated()
    {
        if (isIncapacitated) return;
        
        isIncapacitated = true;
        isAlive = false;
        
        // 전투 불능 대사
        SayDialogue(GetRandomDialogue(DialogueType.Incapacitated), DialogueType.Incapacitated);
        
        OnMechIncapacitated?.Invoke(this);
        Debug.Log($"<color=red>{mechName}이 전투 불능 상태가 되었습니다!</color>");
    }
    
    /// <summary>
    /// 전투 불능 상태에서 회복 가능한지 확인합니다
    /// </summary>
    /// <returns>회복 가능하면 true</returns>
    protected virtual bool CanRecoverFromIncapacitation()
    {
        // 몸통이 파괴되지 않았고, 다른 부위에 최소 HP가 있어야 함
        var torso = bodyParts.FirstOrDefault(bp => bp.partType == BodyPartType.Torso);
        return torso != null && !torso.isDestroyed && stats.currentHP > 0;
    }
    
    /// <summary>
    /// 전투 불능 상태에서 회복합니다
    /// </summary>
    protected virtual void RecoverFromIncapacitation()
    {
        isIncapacitated = false;
        isAlive = true;
        
        // 회복 대사
        SayDialogue(GetRandomDialogue(DialogueType.Revived), DialogueType.Revived);
        
        OnMechRevived?.Invoke(this);
        Debug.Log($"<color=green>{mechName}이 전투 불능 상태에서 회복했습니다!</color>");
    }
    
    /// <summary>
    /// 상태 업데이트 (쿨다운, 버프/디버프 등)
    /// </summary>
    public virtual void UpdateStatuses()
    {
        // 하위 클래스에서 구현
    }
    
    /// <summary>
    /// 보호 상태 업데이트
    /// </summary>
    protected virtual void UpdateProtectionStatus()
    {
        // Update 메서드에서 호출됨
    }
    
    /// <summary>
    /// 쿨다운 업데이트
    /// </summary>
    /// <param name="deltaTime">경과 시간</param>
    public virtual void UpdateCooldowns(float deltaTime)
    {
        // 하위 클래스에서 특수 능력 쿨다운 등을 구현
    }
    
    /// <summary>
    /// 부위 파괴 이벤트 핸들러
    /// </summary>
    /// <param name="destroyedPart">파괴된 부위</param>
    private void OnBodyPartDestroyed_Handler(BodyPart destroyedPart)
    {
        if (!bodyParts.Contains(destroyedPart)) return;
        
        OnBodyPartDestroyed?.Invoke(this, destroyedPart);
        
        // 부위별 파괴 대사
        string partDestroyedDialogue = destroyedPart.partType switch
        {
            BodyPartType.Head => "시야가... 보이지 않아!",
            BodyPartType.Torso => "코어가... 손상됐어...",
            BodyPartType.RightArm => "주무장을 잃었어!",
            BodyPartType.LeftArm => "보조 시스템이 다운됐어!",
            BodyPartType.Legs => "움직일 수 없어!",
            _ => "시스템 손상!"
        };
        
        SayDialogue(partDestroyedDialogue, DialogueType.PartDestroyed);
        
        // 동료들의 걱정 대사 (30% 확률)
        if (UnityEngine.Random.Range(0f, 1f) < 0.3f)
        {
            var allies = FindObjectsOfType<MechCharacter>().Where(m => m != this && m.isAlive).ToList();
            if (allies.Count > 0)
            {
                var worriedAlly = allies[UnityEngine.Random.Range(0, allies.Count)];
                worriedAlly.SayDialogue($"{mechName}! 괜찮아?", DialogueType.Worried);
            }
        }
    }
    
    protected virtual void OnDestroy()
    {
        // 이벤트 구독 해제
        BodyPart.OnPartDestroyed -= OnBodyPartDestroyed_Handler;
    }

    /// <summary>
    /// 현재 행동 가능한지 여부를 반환합니다
    /// </summary>
    public bool CanAct()
    {
        return isAlive && actionPoints != null && actionPoints.currentAP > 0;
    }

    /// <summary>
    /// AP를 소모합니다
    /// </summary>
    public bool ConsumeAP(int amount)
    {
        return actionPoints != null && actionPoints.ConsumeAP(amount);
    }

    /// <summary>
    /// 대화를 트리거합니다 (컨텍스트 문자열은 로깅 용도)
    /// </summary>
    public void TriggerDialogue(string context, string text)
    {
        SayDialogue(text, DialogueType.Cooperative);
    }
}

/// <summary>
/// 기계 타입 열거형
/// </summary>
public enum MechType
{
    Rex,    // 프론트라인 가디언 (탱커)
    Luna,   // 테크니컬 서포터 (힐러/해커)
    Zero,   // 스피드 스카우트 (정찰/기동)
    Nova    // 헤비 어태커 (광역 딜러)
}

/// <summary>
/// 기계 스탯 구조체
/// </summary>
[System.Serializable]
public struct MechStats
{
    public int maxHP;
    public int currentHP;
    public int attack;
    public int defense;
    public int speed;
    public int accuracy;
    public int evasion;
}

/// <summary>
/// 대사 라인 구조체
/// </summary>
[System.Serializable]
public struct DialogueLine
{
    public DialogueType type;
    public string text;
    public float weight; // 선택 확률 가중치
}

/// <summary>
/// 대사 타입 열거형
/// </summary>
public enum DialogueType
{
    TurnStart,          // 턴 시작
    Damage,             // 피해 받을 때
    SevereDamage,       // 심각한 피해 받을 때
    Healed,             // 치료 받을 때
    PartDestroyed,      // 부위 파괴 시
    Incapacitated,      // 전투 불능 시
    Revived,            // 회복 시
    Cooperative,        // 협력 행동 시
    Protective,         // 동료 보호 시
    Grateful,           // 감사 표현 시
    Worried,            // 동료 걱정 시
    Victory,            // 승리 시
    Defeat,             // 패배 시
    Determined,         // 굳은 결의 시
    Critical            // 위기 상황 시
}

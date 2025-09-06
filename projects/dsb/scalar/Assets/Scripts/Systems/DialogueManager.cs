using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// 대화 시스템 매니저
/// 기계들 간의 상호작용, 상황별 대사, 그리고 감정적 반응을 관리합니다.
/// </summary>
public class DialogueManager : MonoBehaviour
{
    [Header("대화 설정")]
    public bool enableDialogues = true;
    public float dialogueDisplayTime = 3f;      // 대사 표시 시간
    public float dialogueDelay = 1f;            // 대사 간 딜레이
    public int maxSimultaneousDialogues = 2;    // 동시 표시 가능한 대사 수
    
    [Header("대화 확률")]
    [Range(0f, 1f)] public float battleStartChance = 0.8f;         // 전투 시작 시
    [Range(0f, 1f)] public float damageReactionChance = 0.4f;      // 피해 받을 때
    [Range(0f, 1f)] public float allyHelpChance = 0.6f;            // 동료 도움 시
    [Range(0f, 1f)] public float victoryChance = 0.9f;             // 승리 시
    [Range(0f, 1f)] public float defeatChance = 0.7f;              // 패배 시
    
    [Header("대화 큐")]
    public Queue<DialogueEntry> dialogueQueue;
    public List<DialogueEntry> activeDialogues;
    
    private static DialogueManager instance;
    public static DialogueManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<DialogueManager>();
                if (instance == null)
                {
                    GameObject go = new GameObject("DialogueManager");
                    instance = go.AddComponent<DialogueManager>();
                }
            }
            return instance;
        }
    }
    
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
            
            dialogueQueue = new Queue<DialogueEntry>();
            activeDialogues = new List<DialogueEntry>();
        }
        else if (instance != this)
        {
            Destroy(gameObject);
        }
    }
    
    private void Start()
    {
        // 이벤트 구독
        MechCharacter.OnDialogueTriggered += OnDialogueTriggered;
        MechCharacter.OnMechIncapacitated += OnMechIncapacitated;
        MechCharacter.OnHealthChanged += OnHealthChanged;
        CooperativeAction.OnCooperativeActionPerformed += OnCooperativeActionPerformed;
        
        StartCoroutine(ProcessDialogueQueue());
        
        Debug.Log("대화 시스템 초기화 완료!");
    }
    
    /// <summary>
    /// 대사가 트리거될 때 호출되는 이벤트 핸들러
    /// </summary>
    private void OnDialogueTriggered(MechCharacter speaker, string dialogue, DialogueType type)
    {
        if (!enableDialogues || string.IsNullOrEmpty(dialogue)) return;
        
        var entry = new DialogueEntry
        {
            speaker = speaker,
            text = dialogue,
            type = type,
            timestamp = Time.time,
            priority = GetDialoguePriority(type)
        };
        
        dialogueQueue.Enqueue(entry);
    }
    
    /// <summary>
    /// 기계가 전투 불능이 되었을 때 동료들의 반응
    /// </summary>
    private void OnMechIncapacitated(MechCharacter incapacitatedMech)
    {
        if (!enableDialogues) return;
        
        StartCoroutine(TriggerAllyReactions(incapacitatedMech, DialogueType.Worried));
    }
    
    /// <summary>
    /// HP 변화 시 반응 (치료/피해)
    /// </summary>
    private void OnHealthChanged(MechCharacter mech, int change)
    {
        if (!enableDialogues) return;
        
        if (change < 0) // 피해
        {
            float damageRatio = Mathf.Abs(change) / (float)mech.stats.maxHP;
            if (damageRatio > 0.2f && Random.Range(0f, 1f) < damageReactionChance)
            {
                // 동료들이 걱정하는 반응
                StartCoroutine(TriggerAllyReactions(mech, DialogueType.Worried, 0.3f));
            }
        }
        else // 치료
        {
            if (change > 0 && Random.Range(0f, 1f) < allyHelpChance)
            {
                // 치료받은 기계의 감사 인사는 이미 MechCharacter에서 처리됨
                // 여기서는 추가적인 동료 반응만 처리
                StartCoroutine(TriggerAllyReactions(mech, DialogueType.Cooperative, 0.2f));
            }
        }
    }
    
    /// <summary>
    /// 협력 행동 수행 시 반응
    /// </summary>
    private void OnCooperativeActionPerformed(CooperativeAction action, List<MechCharacter> actors)
    {
        if (!enableDialogues || actors.Count < 2) return;
        
        // 협력 행동을 보고 다른 동료들의 반응
        var witnesses = FindObjectsOfType<MechCharacter>()
            .Where(m => m.isAlive && !actors.Contains(m)).ToList();
        
        foreach (var witness in witnesses)
        {
            if (Random.Range(0f, 1f) < 0.3f) // 30% 확률
            {
                string reactionDialogue = GetCooperativeReactionDialogue(action.actionName, witness);
                if (!string.IsNullOrEmpty(reactionDialogue))
                {
                    witness.SayDialogue(reactionDialogue, DialogueType.Cooperative);
                }
            }
        }
    }
    
    /// <summary>
    /// 대화 큐를 처리하는 코루틴
    /// </summary>
    private IEnumerator ProcessDialogueQueue()
    {
        while (true)
        {
            // 활성화된 대화 정리 (시간 만료된 것들)
            activeDialogues.RemoveAll(d => Time.time - d.timestamp > dialogueDisplayTime);
            
            // 새 대화 처리
            if (dialogueQueue.Count > 0 && activeDialogues.Count < maxSimultaneousDialogues)
            {
                var nextDialogue = dialogueQueue.Dequeue();
                
                // 중복 대사 체크 (같은 화자가 같은 타입의 대사를 연속으로 하는 것 방지)
                bool isDuplicate = activeDialogues.Any(d => 
                    d.speaker == nextDialogue.speaker && 
                    d.type == nextDialogue.type &&
                    Time.time - d.timestamp < 2f);
                
                if (!isDuplicate)
                {
                    activeDialogues.Add(nextDialogue);
                    DisplayDialogue(nextDialogue);
                }
            }
            
            yield return new WaitForSeconds(0.1f);
        }
    }
    
    /// <summary>
    /// 대사를 화면에 표시합니다
    /// </summary>
    private void DisplayDialogue(DialogueEntry entry)
    {
        // UI가 있다면 UI에 표시, 없으면 Debug.Log로 출력
        string colorCode = GetDialogueColor(entry.type);
        string formattedDialogue = $"<color={colorCode}>[{entry.speaker.mechName}]</color> {entry.text}";
        
        Debug.Log(formattedDialogue);
        
        // 실제 게임에서는 UI 시스템과 연동
        // UIManager.Instance.ShowDialogue(entry);
    }
    
    /// <summary>
    /// 동료들의 반응을 트리거하는 코루틴
    /// </summary>
    private IEnumerator TriggerAllyReactions(MechCharacter targetMech, DialogueType reactionType, float chance = 0.5f)
    {
        yield return new WaitForSeconds(Random.Range(0.5f, 1.5f)); // 자연스러운 딜레이
        
        var allies = FindObjectsOfType<MechCharacter>()
            .Where(m => m != targetMech && m.isAlive).ToList();
        
        // 가장 가까운 동료 또는 랜덤 동료가 반응
        if (allies.Count > 0 && Random.Range(0f, 1f) < chance)
        {
            MechCharacter reactor;
            
            // 거리 기반으로 반응할 동료 선택 (70% 확률로 가장 가까운 동료)
            if (Random.Range(0f, 1f) < 0.7f)
            {
                reactor = allies.OrderBy(a => 
                    Vector3.Distance(a.transform.position, targetMech.transform.position)).First();
            }
            else
            {
                reactor = allies[Random.Range(0, allies.Count)];
            }
            
            string reactionDialogue = GetContextualReaction(reactor, targetMech, reactionType);
            if (!string.IsNullOrEmpty(reactionDialogue))
            {
                reactor.SayDialogue(reactionDialogue, reactionType);
            }
        }
    }
    
    /// <summary>
    /// 상황에 맞는 반응 대사를 생성합니다
    /// </summary>
    private string GetContextualReaction(MechCharacter reactor, MechCharacter target, DialogueType type)
    {
        switch (type)
        {
            case DialogueType.Worried:
                return reactor.mechType switch
                {
                    MechType.Rex => $"{target.mechName}! 괜찮아? 내가 지켜줄게!",
                    MechType.Luna => $"{target.mechName}, 시스템 상태가 위험해! 치료가 필요해!",
                    MechType.Zero => $"{target.mechName}! 빨리 안전한 곳으로 이동해!",
                    MechType.Nova => $"{target.mechName}! 뒤로 물러나! 내가 엄호할게!",
                    _ => $"{target.mechName}! 조심해!"
                };
                
            case DialogueType.Cooperative:
                return reactor.mechType switch
                {
                    MechType.Rex => "좋은 팀워크야!",
                    MechType.Luna => "완벽한 협력이었어!",
                    MechType.Zero => "멋진 연계였어!",
                    MechType.Nova => "이런 협력이 바로 필요한 거야!",
                    _ => "훌륭한 협력이야!"
                };
                
            case DialogueType.Grateful:
                return $"{target.mechName}, 정말 고마워!";
                
            default:
                return "";
        }
    }
    
    /// <summary>
    /// 협력 행동에 대한 반응 대사를 생성합니다
    /// </summary>
    private string GetCooperativeReactionDialogue(string actionName, MechCharacter witness)
    {
        return actionName switch
        {
            "보호하기" => witness.mechType switch
            {
                MechType.Luna => "렉스의 방어가 정말 든든해!",
                MechType.Zero => "렉스 덕분에 안전하게 움직일 수 있어!",
                MechType.Nova => "좋은 방어다! 나도 화력 지원하겠어!",
                _ => "멋진 보호야!"
            },
            "연계 공격" => "완벽한 연계 공격이었어!",
            "응급 처치" => "루나의 치료 실력은 정말 대단해!",
            "전술 이동" => "빠른 판단이었어!",
            _ => "훌륭한 협력이야!"
        };
    }
    
    /// <summary>
    /// 대사 타입에 따른 우선순위 반환
    /// </summary>
    private int GetDialoguePriority(DialogueType type)
    {
        return type switch
        {
            DialogueType.Incapacitated => 10,   // 최고 우선순위
            DialogueType.PartDestroyed => 9,
            DialogueType.SevereDamage => 8,
            DialogueType.Critical => 8,
            DialogueType.Worried => 7,
            DialogueType.Cooperative => 6,
            DialogueType.Protective => 6,
            DialogueType.Victory => 5,
            DialogueType.Defeat => 5,
            DialogueType.Grateful => 4,
            DialogueType.Healed => 3,
            DialogueType.Damage => 2,
            DialogueType.TurnStart => 1,
            _ => 0
        };
    }
    
    /// <summary>
    /// 대사 타입에 따른 색상 코드 반환
    /// </summary>
    private string GetDialogueColor(DialogueType type)
    {
        return type switch
        {
            DialogueType.Incapacitated => "red",
            DialogueType.PartDestroyed => "red",
            DialogueType.SevereDamage => "orange",
            DialogueType.Critical => "red",
            DialogueType.Worried => "yellow",
            DialogueType.Cooperative => "green",
            DialogueType.Protective => "blue",
            DialogueType.Victory => "green",
            DialogueType.Defeat => "gray",
            DialogueType.Grateful => "cyan",
            DialogueType.Healed => "green",
            _ => "white"
        };
    }
    
    /// <summary>
    /// 전투 시작 시 팀 대화 시퀀스
    /// </summary>
    public IEnumerator PlayBattleStartDialogue(List<MechCharacter> team)
    {
        if (!enableDialogues || team.Count == 0) yield break;
        
        yield return new WaitForSeconds(1f);
        
        // 팀 리더 (첫 번째 기계)가 먼저 말함
        var leader = team[0];
        leader.SayDialogue(leader.GetRandomDialogue(DialogueType.TurnStart), DialogueType.TurnStart);
        
        yield return new WaitForSeconds(2f);
        
        // 다른 팀원들이 순서대로 반응
        for (int i = 1; i < team.Count && i < 3; i++) // 최대 3명만
        {
            if (Random.Range(0f, 1f) < battleStartChance)
            {
                team[i].SayDialogue(team[i].GetRandomDialogue(DialogueType.TurnStart), DialogueType.TurnStart);
                yield return new WaitForSeconds(1.5f);
            }
        }
    }
    
    /// <summary>
    /// 특별한 상황에서의 대화 트리거
    /// </summary>
    public void TriggerSpecialDialogue(string situation, MechCharacter speaker = null)
    {
        if (!enableDialogues) return;
        
        switch (situation)
        {
            case "first_battle":
                if (speaker != null)
                {
                    speaker.SayDialogue("첫 전투다... 모두 조심하자!", DialogueType.TurnStart);
                }
                break;
                
            case "last_stand":
                var survivors = FindObjectsOfType<MechCharacter>().Where(m => m.isAlive).ToList();
                if (survivors.Count == 1)
                {
                    survivors[0].SayDialogue("나 혼자 남았지만... 포기하지 않을 거야!", DialogueType.Critical);
                }
                break;
                
            case "perfect_victory":
                var aliveMechs = FindObjectsOfType<MechCharacter>().Where(m => m.isAlive).ToList();
                if (aliveMechs.Count > 0)
                {
                    var celebrate = aliveMechs[Random.Range(0, aliveMechs.Count)];
                    celebrate.SayDialogue("완벽한 승리야! 아무도 다치지 않았어!", DialogueType.Victory);
                }
                break;
        }
    }
    
    /// <summary>
    /// 대화 시스템 정리
    /// </summary>
    public void ClearAllDialogues()
    {
        dialogueQueue.Clear();
        activeDialogues.Clear();
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        MechCharacter.OnDialogueTriggered -= OnDialogueTriggered;
        MechCharacter.OnMechIncapacitated -= OnMechIncapacitated;
        MechCharacter.OnHealthChanged -= OnHealthChanged;
        CooperativeAction.OnCooperativeActionPerformed -= OnCooperativeActionPerformed;
    }
}

/// <summary>
/// 대화 엔트리 클래스
/// </summary>
[System.Serializable]
public class DialogueEntry
{
    public MechCharacter speaker;
    public string text;
    public DialogueType type;
    public float timestamp;
    public int priority;
    
    public override string ToString()
    {
        return $"[{speaker?.mechName}] {text} ({type})";
    }
}

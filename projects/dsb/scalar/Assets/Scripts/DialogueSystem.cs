using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class DialogueSystem : MonoBehaviour
{
    [Header("대사 시스템")]
    public static DialogueSystem Instance;
    
    [Header("UI")]
    public GameObject dialoguePanel;
    public UnityEngine.UI.Text dialogueText;
    public UnityEngine.UI.Text speakerName;
    public float displayDuration = 3f;
    
    [Header("대사 데이터")]
    public DialogueDatabase dialogueDatabase;
    
    private Queue<DialogueEntry> dialogueQueue = new Queue<DialogueEntry>();
    private bool isDisplaying = false;
    
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    private void Start()
    {
        InitializeDialogueSystem();
        SubscribeToEvents();
    }
    
    private void InitializeDialogueSystem()
    {
        if (dialogueDatabase == null)
        {
            dialogueDatabase = ScriptableObject.CreateInstance<DialogueDatabase>();
        }
        
        if (dialoguePanel != null)
        {
            dialoguePanel.SetActive(false);
        }
    }
    
    private void SubscribeToEvents()
    {
        MechCharacter.OnDialogueTriggered += OnDialogueTriggered;
        CooperationSystem.OnCooperationUsed += OnCooperationUsed;
        CooperationSystem.OnTrustIncreased += OnTrustIncreased;
    }
    
    private void OnDestroy()
    {
        MechCharacter.OnDialogueTriggered -= OnDialogueTriggered;
        CooperationSystem.OnCooperationUsed -= OnCooperationUsed;
        CooperationSystem.OnTrustIncreased -= OnTrustIncreased;
    }
    
    private void OnDialogueTriggered(MechCharacter mech, string dialogue)
    {
        ShowDialogue(mech.mechName, dialogue, DialogueType.General);
    }
    
    private void OnCooperationUsed(MechCharacter user, MechCharacter target, string skillName)
    {
        string dialogue = GetCooperationDialogue(user, target, skillName);
        ShowDialogue(user.mechName, dialogue, DialogueType.Cooperation);
    }
    
    private void OnTrustIncreased(MechCharacter user, MechCharacter target, int trustGain)
    {
        if (trustGain >= 10)
        {
            string dialogue = GetTrustDialogue(user, target);
            ShowDialogue(user.mechName, dialogue, DialogueType.Trust);
        }
    }
    
    public void ShowDialogue(string speaker, string dialogue, DialogueType type)
    {
        DialogueEntry entry = new DialogueEntry
        {
            speaker = speaker,
            dialogue = dialogue,
            type = type
        };
        
        dialogueQueue.Enqueue(entry);
        
        if (!isDisplaying)
        {
            StartCoroutine(DisplayNextDialogue());
        }
    }
    
    private IEnumerator DisplayNextDialogue()
    {
        isDisplaying = true;
        
        while (dialogueQueue.Count > 0)
        {
            DialogueEntry entry = dialogueQueue.Dequeue();
            
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(true);
                
                if (speakerName != null)
                {
                    speakerName.text = entry.speaker;
                }
                
                if (dialogueText != null)
                {
                    dialogueText.text = entry.dialogue;
                }
                
                // 대사 타입에 따른 색상 변경
                SetDialogueColor(entry.type);
            }
            
            yield return new WaitForSeconds(displayDuration);
            
            if (dialoguePanel != null)
            {
                dialoguePanel.SetActive(false);
            }
        }
        
        isDisplaying = false;
    }
    
    private void SetDialogueColor(DialogueType type)
    {
        if (dialogueText == null) return;
        
        switch (type)
        {
            case DialogueType.General:
                dialogueText.color = Color.white;
                break;
            case DialogueType.Cooperation:
                dialogueText.color = Color.cyan;
                break;
            case DialogueType.Trust:
                dialogueText.color = Color.yellow;
                break;
            case DialogueType.Damage:
                dialogueText.color = Color.red;
                break;
            case DialogueType.Victory:
                dialogueText.color = Color.green;
                break;
            case DialogueType.Defeat:
                dialogueText.color = Color.gray;
                break;
        }
    }
    
    private string GetCooperationDialogue(MechCharacter user, MechCharacter target, string skillName)
    {
        // 협력 스킬에 따른 대사
        switch (skillName)
        {
            case "가드":
                return $"{target.mechName}을 지켜줄게!";
            case "응급처치":
                return $"{target.mechName}, 괜찮아질 거야!";
            case "전술이동":
                return "위치 바꿔!";
            case "연계공격":
                return $"{target.mechName}과 함께!";
            case "지원 사격":
                return $"{target.mechName}, 지원할게!";
            case "합동 방어":
                return "함께 버티자!";
            default:
                return "협력하자!";
        }
    }
    
    private string GetTrustDialogue(MechCharacter user, MechCharacter target)
    {
        // 신뢰도 증가에 따른 대사
        int trustLevel = user.trustLevels.ContainsKey(target.mechType) ? user.trustLevels[target.mechType] : 0;
        
        if (trustLevel >= 100)
        {
            return $"{target.mechName}과는 이제 진짜 팀이야!";
        }
        else if (trustLevel >= 50)
        {
            return $"{target.mechName}과의 팀워크가 좋아지고 있어!";
        }
        else
        {
            return $"{target.mechName}을 믿을 수 있겠어!";
        }
    }
    
    public void ShowBattleStartDialogue()
    {
        string[] startDialogues = {
            "전투 시작! 모두 조심해!",
            "우리 팀워크를 보여주자!",
            "모두 함께 집에 돌아가자!",
            "동료들을 지켜야 해!"
        };
        
        string dialogue = startDialogues[Random.Range(0, startDialogues.Length)];
        ShowDialogue("팀", dialogue, DialogueType.General);
    }
    
    public void ShowVictoryDialogue()
    {
        string[] victoryDialogues = {
            "승리다! 모두 잘했어!",
            "우리 팀워크가 승리를 가져왔어!",
            "아무도 다치지 않고 승리했어!",
            "진짜 승리는 모두 함께 집에 돌아가는 거야!"
        };
        
        string dialogue = victoryDialogues[Random.Range(0, victoryDialogues.Length)];
        ShowDialogue("팀", dialogue, DialogueType.Victory);
    }
    
    public void ShowDefeatDialogue()
    {
        string[] defeatDialogues = {
            "후퇴하자... 다음엔 더 잘할 수 있어.",
            "동료들을 구해야 해...",
            "이번엔 실패했지만, 포기하지 않을 거야.",
            "다시 정비하고 돌아오자."
        };
        
        string dialogue = defeatDialogues[Random.Range(0, defeatDialogues.Length)];
        ShowDialogue("팀", dialogue, DialogueType.Defeat);
    }
    
    public void ShowRetreatDialogue()
    {
        string[] retreatDialogues = {
            "전략적 후퇴다!",
            "이 상황에서는 후퇴가 현명해.",
            "다시 정비하고 돌아오자.",
            "무리하지 말고 후퇴하자."
        };
        
        string dialogue = retreatDialogues[Random.Range(0, retreatDialogues.Length)];
        ShowDialogue("팀", dialogue, DialogueType.General);
    }
}

[System.Serializable]
public class DialogueEntry
{
    public string speaker;
    public string dialogue;
    public DialogueType type;
}

public enum DialogueType
{
    General,        // 일반 대사
    Cooperation,    // 협력 대사
    Trust,         // 신뢰도 관련 대사
    Damage,        // 피해 관련 대사
    Victory,       // 승리 대사
    Defeat         // 패배 대사
}

[CreateAssetMenu(fileName = "DialogueDatabase", menuName = "Scalar/Dialogue Database")]
public class DialogueDatabase : ScriptableObject
{
    [Header("대사 데이터")]
    public List<DialogueData> dialogues = new List<DialogueData>();
    
    public string GetDialogue(string situation, MechType mechType)
    {
        foreach (DialogueData data in dialogues)
        {
            if (data.situation == situation && data.mechType == mechType)
            {
                if (data.dialogues.Count > 0)
                {
                    return data.dialogues[Random.Range(0, data.dialogues.Count)];
                }
            }
        }
        
        return "대사가 없습니다.";
    }
}

[System.Serializable]
public class DialogueData
{
    public string situation;
    public MechType mechType;
    public List<string> dialogues = new List<string>();
}

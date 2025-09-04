using UnityEngine;
using System.Collections.Generic;

public class GameManager : MonoBehaviour
{
    [Header("게임 매니저")]
    public static GameManager Instance;
    
    [Header("시스템 참조")]
    public BattleSystem battleSystem;
    public CooperationSystem cooperationSystem;
    public DialogueSystem dialogueSystem;
    public BattleUI battleUI;
    
    [Header("게임 상태")]
    public GameState currentState = GameState.Exploration;
    public bool isGamePaused = false;
    
    [Header("플레이어 팀")]
    public List<MechCharacter> playerTeam = new List<MechCharacter>();
    
    [Header("게임 설정")]
    public bool enableCooperation = true;
    public bool enableDialogue = true;
    public bool enableRetreat = true;
    
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    private void Start()
    {
        InitializeGame();
    }
    
    private void InitializeGame()
    {
        // 시스템 초기화
        InitializeSystems();
        
        // 플레이어 팀 초기화
        InitializePlayerTeam();
        
        // 이벤트 구독
        SubscribeToEvents();
        
        Debug.Log("게임이 초기화되었습니다.");
    }
    
    private void InitializeSystems()
    {
        // 시스템들 찾기
        if (battleSystem == null)
            battleSystem = FindObjectOfType<BattleSystem>();
        
        if (cooperationSystem == null)
            cooperationSystem = FindObjectOfType<CooperationSystem>();
        
        if (dialogueSystem == null)
            dialogueSystem = FindObjectOfType<DialogueSystem>();
        
        if (battleUI == null)
            battleUI = FindObjectOfType<BattleUI>();
    }
    
    private void InitializePlayerTeam()
    {
        // 플레이어 기계들 찾기
        MechCharacter[] mechs = FindObjectsOfType<MechCharacter>();
        playerTeam.Clear();
        
        foreach (MechCharacter mech in mechs)
        {
            if (mech.isAlive)
            {
                playerTeam.Add(mech);
            }
        }
        
        Debug.Log($"플레이어 팀 초기화 완료: {playerTeam.Count}명");
    }
    
    private void SubscribeToEvents()
    {
        BattleSystem.OnBattleStart += OnBattleStart;
        BattleSystem.OnBattleEnd += OnBattleEnd;
    }
    
    private void OnDestroy()
    {
        BattleSystem.OnBattleStart -= OnBattleStart;
        BattleSystem.OnBattleEnd -= OnBattleEnd;
    }
    
    private void OnBattleStart(BattleSystem battle)
    {
        currentState = GameState.Battle;
        Debug.Log("전투 모드로 전환");
        
        if (dialogueSystem != null)
        {
            dialogueSystem.ShowBattleStartDialogue();
        }
    }
    
    private void OnBattleEnd(BattleSystem battle)
    {
        currentState = GameState.Exploration;
        Debug.Log("탐험 모드로 전환");
        
        // 전투 결과에 따른 처리
        ProcessBattleResult(battle);
    }
    
    private void ProcessBattleResult(BattleSystem battle)
    {
        // 승리/패배 확인
        bool victory = battle.GetAliveEnemies().Count == 0;
        
        if (victory)
        {
            Debug.Log("전투 승리!");
            if (dialogueSystem != null)
            {
                dialogueSystem.ShowVictoryDialogue();
            }
            
            // 승리 보상 처리
            ProcessVictoryRewards();
        }
        else
        {
            Debug.Log("전투 패배 또는 후퇴");
            if (dialogueSystem != null)
            {
                dialogueSystem.ShowDefeatDialogue();
            }
            
            // 패배 처리
            ProcessDefeatConsequences();
        }
    }
    
    private void ProcessVictoryRewards()
    {
        // 승리 보상 처리
        foreach (MechCharacter mech in playerTeam)
        {
            if (mech.isAlive)
            {
                // 경험치, 아이템, 신뢰도 증가 등
                Debug.Log($"{mech.mechName}이 승리 보상을 획득했습니다.");
            }
        }
    }
    
    private void ProcessDefeatConsequences()
    {
        // 패배 결과 처리
        foreach (MechCharacter mech in playerTeam)
        {
            if (!mech.isAlive)
            {
                Debug.Log($"{mech.mechName}이 전투 불능 상태입니다.");
                // 드라이브 회수 미션 등
            }
        }
    }
    
    public void StartBattle()
    {
        if (battleSystem != null && currentState == GameState.Exploration)
        {
            battleSystem.StartBattle();
        }
    }
    
    public void PauseGame()
    {
        isGamePaused = true;
        Time.timeScale = 0f;
        Debug.Log("게임 일시정지");
    }
    
    public void ResumeGame()
    {
        isGamePaused = false;
        Time.timeScale = 1f;
        Debug.Log("게임 재개");
    }
    
    public void RetreatFromBattle()
    {
        if (battleSystem != null && currentState == GameState.Battle)
        {
            battleSystem.AttemptRetreat();
        }
    }
    
    public List<MechCharacter> GetAlivePlayerMechs()
    {
        List<MechCharacter> aliveMechs = new List<MechCharacter>();
        
        foreach (MechCharacter mech in playerTeam)
        {
            if (mech.isAlive)
            {
                aliveMechs.Add(mech);
            }
        }
        
        return aliveMechs;
    }
    
    public bool IsTeamAlive()
    {
        return GetAlivePlayerMechs().Count > 0;
    }
    
    public void SaveGame()
    {
        // 게임 저장 로직
        Debug.Log("게임 저장");
    }
    
    public void LoadGame()
    {
        // 게임 로드 로직
        Debug.Log("게임 로드");
    }
}

public enum GameState
{
    Exploration,    // 탐험 모드
    Battle,         // 전투 모드
    Menu,           // 메뉴 모드
    Paused          // 일시정지 모드
}

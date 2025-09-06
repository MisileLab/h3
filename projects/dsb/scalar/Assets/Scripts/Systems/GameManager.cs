using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Scalar 게임의 메인 게임 매니저
/// 전투 시스템, 대화 시스템, UI 등을 총괄 관리합니다.
/// </summary>
public class GameManager : MonoBehaviour
{
    [Header("게임 상태")]
    public GameState currentState = GameState.Menu;
    public bool isGamePaused = false;
    
    [Header("시스템 매니저들")]
    public BattleSystem battleSystem;
    public DialogueManager dialogueManager;
    public CooperativeActionManager cooperativeManager;
    public BattleInitializer battleInitializer;
    
    [Header("게임 설정")]
    public bool autoStartBattle = true;
    public float gameStartDelay = 2f;
    public bool enableTutorial = false;
    
    [Header("디버그")]
    public bool debugMode = false;
    public bool showSystemStatus = true;
    
    private static GameManager instance;
    public static GameManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<GameManager>();
                if (instance == null)
                {
                    GameObject go = new GameObject("GameManager");
                    instance = go.AddComponent<GameManager>();
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
        }
        else if (instance != this)
        {
            Destroy(gameObject);
            return;
        }
        
        InitializeGame();
    }
    
    private void Start()
    {
        StartCoroutine(GameStartSequence());
    }
    
    private void Update()
    {
        HandleInput();
        
        if (debugMode && showSystemStatus)
        {
            UpdateDebugInfo();
        }
    }
    
    /// <summary>
    /// 게임 초기화
    /// </summary>
    private void InitializeGame()
    {
        Debug.Log("=== Scalar 게임 시스템 초기화 ===");
        
        // 시스템 매니저들 찾기 또는 생성
        InitializeSystemManagers();
        
        // 이벤트 구독
        SubscribeToEvents();
        
        Debug.Log("게임 시스템 초기화 완료!");
        Debug.Log($"핵심 철학: \"{GetGamePhilosophy()}\"");
    }
    
    /// <summary>
    /// 시스템 매니저들 초기화
    /// </summary>
    private void InitializeSystemManagers()
    {
        // BattleSystem 찾기 또는 생성
        if (battleSystem == null)
        {
            battleSystem = FindObjectOfType<BattleSystem>();
            if (battleSystem == null)
            {
                GameObject battleGO = new GameObject("BattleSystem");
                battleSystem = battleGO.AddComponent<BattleSystem>();
            }
        }
        
        // DialogueManager 확인
        if (dialogueManager == null)
        {
            dialogueManager = DialogueManager.Instance;
        }
        
        // CooperativeActionManager 찾기 또는 생성
        if (cooperativeManager == null)
        {
            cooperativeManager = FindObjectOfType<CooperativeActionManager>();
            if (cooperativeManager == null)
            {
                GameObject coopGO = new GameObject("CooperativeActionManager");
                cooperativeManager = coopGO.AddComponent<CooperativeActionManager>();
            }
        }
        
        // BattleInitializer 찾기
        if (battleInitializer == null)
        {
            battleInitializer = FindObjectOfType<BattleInitializer>();
        }
        
        Debug.Log("시스템 매니저들 초기화 완료");
    }
    
    /// <summary>
    /// 이벤트 구독
    /// </summary>
    private void SubscribeToEvents()
    {
        if (battleSystem != null)
        {
            BattleSystem.OnBattleStart += OnBattleStart;
            BattleSystem.OnBattleEnd += OnBattleEnd;
        }
        
        MechCharacter.OnMechIncapacitated += OnMechIncapacitated;
        MechCharacter.OnMechRevived += OnMechRevived;
    }
    
    /// <summary>
    /// 게임 시작 시퀀스
    /// </summary>
    private IEnumerator GameStartSequence()
    {
        currentState = GameState.Loading;
        
        yield return new WaitForSeconds(gameStartDelay);
        
        if (enableTutorial)
        {
            yield return StartCoroutine(PlayTutorial());
        }
        
        if (autoStartBattle)
        {
            StartBattle();
        }
        else
        {
            currentState = GameState.Menu;
            Debug.Log("게임 준비 완료! 전투를 시작하려면 StartBattle()을 호출하세요.");
        }
    }
    
    /// <summary>
    /// 튜토리얼 재생
    /// </summary>
    private IEnumerator PlayTutorial()
    {
        currentState = GameState.Tutorial;
        Debug.Log("=== Scalar 튜토리얼 시작 ===");
        
        yield return new WaitForSeconds(1f);
        
        Debug.Log("Scalar는 협력형 턴제 전략 RPG입니다.");
        yield return new WaitForSeconds(2f);
        
        Debug.Log("핵심은 '보호와 협력'입니다. 적을 제거하는 것보다 동료를 지키는 것이 더 중요합니다.");
        yield return new WaitForSeconds(3f);
        
        Debug.Log("각 기계는 고유한 능력을 가지고 있으며, 협력 행동을 통해 더 강해집니다.");
        yield return new WaitForSeconds(3f);
        
        Debug.Log("진정한 승리는 모두 함께 집에 돌아가는 것입니다!");
        yield return new WaitForSeconds(2f);
        
        Debug.Log("=== 튜토리얼 완료 ===");
    }
    
    /// <summary>
    /// 전투 시작
    /// </summary>
    public void StartBattle()
    {
        if (currentState == GameState.Battle)
        {
            Debug.LogWarning("이미 전투가 진행 중입니다.");
            return;
        }
        
        Debug.Log("=== 전투 준비 ===");
        currentState = GameState.Preparation;
        
        // 전투 초기화
        if (battleInitializer != null)
        {
            battleInitializer.InitializeAndStartBattle();
        }
        else if (battleSystem != null)
        {
            battleSystem.StartBattle();
        }
        else
        {
            Debug.LogError("BattleSystem을 찾을 수 없습니다!");
        }
    }
    
    /// <summary>
    /// 전투 일시정지/재개
    /// </summary>
    public void TogglePause()
    {
        isGamePaused = !isGamePaused;
        Time.timeScale = isGamePaused ? 0f : 1f;
        
        Debug.Log(isGamePaused ? "게임 일시정지" : "게임 재개");
    }
    
    /// <summary>
    /// 입력 처리
    /// </summary>
    private void HandleInput()
    {
        // ESC 키로 일시정지
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            if (currentState == GameState.Battle)
            {
                TogglePause();
            }
        }
        
        // 디버그 모드 토글 (F12)
        if (Input.GetKeyDown(KeyCode.F12))
        {
            debugMode = !debugMode;
            Debug.Log($"디버그 모드: {(debugMode ? "활성화" : "비활성화")}");
        }
        
        // 전투 수동 시작 (스페이스바)
        if (Input.GetKeyDown(KeyCode.Space) && currentState == GameState.Menu)
        {
            StartBattle();
        }
        
        // 강제 전투 종료 (F10)
        if (Input.GetKeyDown(KeyCode.F10) && debugMode && currentState == GameState.Battle)
        {
            Debug.Log("[디버그] 강제 전투 종료");
            EndBattle(false);
        }
    }
    
    /// <summary>
    /// 디버그 정보 업데이트
    /// </summary>
    private void UpdateDebugInfo()
    {
        // 3초마다 시스템 상태 출력
        if (Time.time % 3f < 0.1f)
        {
            LogSystemStatus();
        }
    }
    
    /// <summary>
    /// 시스템 상태 로그 출력
    /// </summary>
    private void LogSystemStatus()
    {
        Debug.Log($"=== 시스템 상태 ({Time.time:F1}s) ===");
        Debug.Log($"게임 상태: {currentState}");
        Debug.Log($"전투 활성: {(battleSystem != null ? battleSystem.isBattleActive : false)}");
        Debug.Log($"대화 큐: {(dialogueManager != null ? dialogueManager.dialogueQueue.Count : 0)}개");
        
        if (battleSystem != null && battleSystem.isBattleActive)
        {
            var alivePlayers = battleSystem.GetAlivePlayerMechs().Count;
            var totalPlayers = battleSystem.playerMechs.Count;
            var aliveEnemies = battleSystem.GetAliveEnemies().Count;
            var totalEnemies = battleSystem.enemies.Count;
            
            Debug.Log($"생존 현황: 아군 {alivePlayers}/{totalPlayers}, 적군 {aliveEnemies}/{totalEnemies}");
        }
    }
    
    /// <summary>
    /// 전투 시작 이벤트 핸들러
    /// </summary>
    private void OnBattleStart(BattleSystem battle)
    {
        currentState = GameState.Battle;
        Debug.Log("<color=green>=== 전투 시작! ===</color>");
        
        // 전투 시작 대화 재생
        if (dialogueManager != null)
        {
            var team = battle.GetAlivePlayerMechs();
            StartCoroutine(dialogueManager.PlayBattleStartDialogue(team));
        }
        
        // 첫 전투인지 확인하고 특별 대사
        if (battle.currentTurn == 1)
        {
            dialogueManager?.TriggerSpecialDialogue("first_battle", battle.GetAlivePlayerMechs().FirstOrDefault());
        }
    }
    
    /// <summary>
    /// 전투 종료 이벤트 핸들러
    /// </summary>
    private void OnBattleEnd(BattleSystem battle)
    {
        bool victory = battle.GetAliveEnemies().Count == 0;
        EndBattle(victory);
    }
    
    /// <summary>
    /// 전투 종료 처리
    /// </summary>
    private void EndBattle(bool victory)
    {
        currentState = GameState.Result;
        
        if (victory)
        {
            Debug.Log("<color=green>=== 전투 승리! ===</color>");
            
            // 완벽한 승리인지 확인
            if (battleSystem != null && battleSystem.perfectProtection)
            {
                dialogueManager?.TriggerSpecialDialogue("perfect_victory");
            }
        }
        else
        {
            Debug.Log("<color=yellow>=== 전투 패배 ===</color>");
            Debug.Log("하지만 이는 끝이 아니다. 새로운 시작이다.");
        }
        
        // 결과 화면을 잠시 보여준 후 메뉴로 복귀
        StartCoroutine(ReturnToMenuAfterDelay(5f));
    }
    
    /// <summary>
    /// 딜레이 후 메뉴로 복귀
    /// </summary>
    private IEnumerator ReturnToMenuAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        
        ReturnToMenu();
    }
    
    /// <summary>
    /// 메뉴로 복귀
    /// </summary>
    public void ReturnToMenu()
    {
        currentState = GameState.Menu;
        
        // 대화 정리
        if (dialogueManager != null)
        {
            dialogueManager.ClearAllDialogues();
        }
        
        Debug.Log("메뉴로 돌아갔습니다. 새로운 전투를 시작하려면 StartBattle()을 호출하세요.");
    }
    
    /// <summary>
    /// 기계 전투 불능 이벤트 핸들러
    /// </summary>
    private void OnMechIncapacitated(MechCharacter mech)
    {
        Debug.Log($"<color=red>{mech.mechName}이 전투 불능 상태가 되었습니다.</color>");
        
        // 마지막 한 명이 남았을 때 특별 대사
        if (battleSystem != null)
        {
            var survivors = battleSystem.GetAlivePlayerMechs();
            if (survivors.Count == 1)
            {
                dialogueManager?.TriggerSpecialDialogue("last_stand", survivors[0]);
            }
        }
    }
    
    /// <summary>
    /// 기계 회복 이벤트 핸들러
    /// </summary>
    private void OnMechRevived(MechCharacter mech)
    {
        Debug.Log($"<color=green>{mech.mechName}이 다시 일어났습니다!</color>");
    }
    
    /// <summary>
    /// 게임 철학 반환
    /// </summary>
    public string GetGamePhilosophy()
    {
        return "진정한 승리는 적을 섬멸하는 것이 아니라, 모두 함께 집에 돌아가는 것이다.";
    }
    
    /// <summary>
    /// 게임 종료
    /// </summary>
    public void QuitGame()
    {
        Debug.Log("게임을 종료합니다.");
        
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
    
    /// <summary>
    /// 협력 행동 사용 (외부 호출용)
    /// </summary>
    public bool UseCooperativeAction(string actionName, List<MechCharacter> actors, object target = null)
    {
        if (cooperativeManager == null) return false;
        
        var action = cooperativeManager.availableActions.Find(a => a.actionName == actionName);
        if (action != null)
        {
            return battleSystem.TryCooperativeAction(action, actors, target);
        }
        
        return false;
    }
    
    /// <summary>
    /// 현재 턴인 기계 반환
    /// </summary>
    public MechCharacter GetCurrentMech()
    {
        if (battleSystem == null) return null;
        
        var currentActor = battleSystem.GetCurrentActor();
        return currentActor?.isEnemy == false ? currentActor.mech : null;
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        BattleSystem.OnBattleStart -= OnBattleStart;
        BattleSystem.OnBattleEnd -= OnBattleEnd;
        MechCharacter.OnMechIncapacitated -= OnMechIncapacitated;
        MechCharacter.OnMechRevived -= OnMechRevived;
        
        // 시간 배율 복구
        Time.timeScale = 1f;
    }
}

/// <summary>
/// 게임 상태 열거형
/// </summary>
public enum GameState
{
    Menu,           // 메뉴
    Loading,        // 로딩 중
    Tutorial,       // 튜토리얼
    Preparation,    // 전투 준비
    Battle,         // 전투 중
    Result,         // 결과 화면
    Paused          // 일시정지
}

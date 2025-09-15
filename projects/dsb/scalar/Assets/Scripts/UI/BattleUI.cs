using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class BattleUI : MonoBehaviour
{
    [Header("UI 패널")]
    public GameObject battlePanel;
    public GameObject actionPanel;
    public GameObject cooperationPanel;
    public GameObject statusPanel;
    public GameObject dialoguePanel;
    
    [Header("턴 정보")]
    public Text turnInfoText;
    public Text currentActorText;
    public Slider turnProgressSlider;
    
    [Header("기계 상태")]
    public Transform mechStatusContainer;
    public GameObject mechStatusPrefab;
    private List<MechStatusUI> mechStatusUIs = new List<MechStatusUI>();
    
    [Header("행동 버튼")]
    public Button attackButton;
    public Button defendButton;
    public Button skillButton;
    public Button moveButton;           // 새로 추가된 이동 버튼
    public Button cooperationButton;
    public Button retreatButton;
    public Button endTurnButton;
    
    [Header("협력 UI")]
    public Transform cooperationContainer;
    public GameObject cooperationOptionPrefab;
    
    [Header("대화 UI")]
    public Text dialogueText;
    public Text speakerName;
    public Image dialogueBackground;
    
    [Header("AP 표시")]
    public Text apText;
    public Slider apSlider;
    
    private BattleSystem battleSystem;
    private MechCharacter currentMech;
    private List<MechCharacter> availableTargets = new List<MechCharacter>();
    
    // 새로운 시스템 참조들
    private ActionModeManager actionModeManager;
    private RangeDisplay rangeDisplay;
    private TargetSelector targetSelector;
    
    private void Start()
    {
        InitializeUI();
        SubscribeToEvents();
    }
    
    private void InitializeUI()
    {
        battleSystem = FindObjectOfType<BattleSystem>();
        
        // 새로운 시스템들 초기화
        InitializeNewSystems();
        
        // 버튼 이벤트 연결
        if (attackButton != null) attackButton.onClick.AddListener(OnAttackClicked);
        if (defendButton != null) defendButton.onClick.AddListener(OnDefendClicked);
        if (skillButton != null) skillButton.onClick.AddListener(OnSkillClicked);
        if (moveButton != null) moveButton.onClick.AddListener(OnMoveClicked);        // 이동 버튼 추가
        if (cooperationButton != null) cooperationButton.onClick.AddListener(OnCooperationClicked);
        if (retreatButton != null) retreatButton.onClick.AddListener(OnRetreatClicked);
        if (endTurnButton != null) endTurnButton.onClick.AddListener(OnEndTurnClicked);
        
        // 초기 상태 설정 - 테스트를 위해 UI를 활성화 상태로 유지
        if (battlePanel != null) battlePanel.SetActive(true);   // 전투 패널 활성화
        if (actionPanel != null) actionPanel.SetActive(true);   // 액션 패널 활성화 (테스트용)
        if (cooperationPanel != null) cooperationPanel.SetActive(false); // 협력 패널은 필요시에만
    }
    
    /// <summary>
    /// 새로운 시스템들을 초기화합니다
    /// </summary>
    private void InitializeNewSystems()
    {
        // ActionModeManager 찾기 또는 생성
        actionModeManager = FindObjectOfType<ActionModeManager>();
        if (actionModeManager == null)
        {
            GameObject actionModeObj = new GameObject("ActionModeManager");
            actionModeManager = actionModeObj.AddComponent<ActionModeManager>();
            Debug.Log("ActionModeManager 자동 생성됨");
        }
        
        // RangeDisplay 찾기 또는 생성
        rangeDisplay = FindObjectOfType<RangeDisplay>();
        if (rangeDisplay == null)
        {
            GameObject rangeDisplayObj = new GameObject("RangeDisplay");
            rangeDisplay = rangeDisplayObj.AddComponent<RangeDisplay>();
            Debug.Log("RangeDisplay 자동 생성됨");
        }
        
        // TargetSelector 찾기 또는 생성
        targetSelector = FindObjectOfType<TargetSelector>();
        if (targetSelector == null)
        {
            GameObject targetSelectorObj = new GameObject("TargetSelector");
            targetSelector = targetSelectorObj.AddComponent<TargetSelector>();
            Debug.Log("TargetSelector 자동 생성됨");
        }
        
        Debug.Log("새로운 시스템 초기화 완료");
    }
    
    private void SubscribeToEvents()
    {
        BattleSystem.OnBattleStart += OnBattleStart;
        BattleSystem.OnBattleEnd += OnBattleEnd;
        BattleSystem.OnTurnStart += OnTurnStart;
        BattleSystem.OnTurnEnd += OnTurnEnd;
    }
    
    private void OnDestroy()
    {
        BattleSystem.OnBattleStart -= OnBattleStart;
        BattleSystem.OnBattleEnd -= OnBattleEnd;
        BattleSystem.OnTurnStart -= OnTurnStart;
        BattleSystem.OnTurnEnd -= OnTurnEnd;
    }
    
    private void OnBattleStart(BattleSystem battle)
    {
        if (battlePanel != null) battlePanel.SetActive(true);
        CreateMechStatusUIs();
        UpdateUI();
    }
    
    private void OnBattleEnd(BattleSystem battle)
    {
        if (battlePanel != null) battlePanel.SetActive(false);
        if (actionPanel != null) actionPanel.SetActive(false);
        if (cooperationPanel != null) cooperationPanel.SetActive(false);
    }
    
    private void OnTurnStart(BattleSystem battle, BattleActor actor)
    {
        if (!actor.isEnemy)
        {
            currentMech = actor.mech;
            if (actionPanel != null) actionPanel.SetActive(true);
            UpdateActionButtons();
            UpdateAPDisplay();
        }
        else
        {
            if (actionPanel != null) actionPanel.SetActive(false);
        }
        
        UpdateTurnInfo(actor);
    }
    
    private void OnTurnEnd(BattleSystem battle, BattleActor actor)
    {
        if (cooperationPanel != null) cooperationPanel.SetActive(false);
    }
    
    private void CreateMechStatusUIs()
    {
        // 기존 UI 제거
        foreach (MechStatusUI ui in mechStatusUIs)
        {
            if (ui != null) Destroy(ui.gameObject);
        }
        mechStatusUIs.Clear();
        
        // 새로운 UI 생성
        List<MechCharacter> mechs = battleSystem.GetAlivePlayerMechs();
        foreach (MechCharacter mech in mechs)
        {
            if (mechStatusPrefab != null && mechStatusContainer != null)
            {
                GameObject statusUI = Instantiate(mechStatusPrefab, mechStatusContainer);
                MechStatusUI statusComponent = statusUI.GetComponent<MechStatusUI>();
                if (statusComponent != null)
                {
                    statusComponent.Initialize(mech);
                    mechStatusUIs.Add(statusComponent);
                }
            }
        }
    }
    
    private void UpdateUI()
    {
        UpdateTurnInfo(battleSystem.GetCurrentActor());
        UpdateMechStatusUIs();
    }
    
    private void UpdateTurnInfo(BattleActor actor)
    {
        if (actor != null)
        {
            if (currentActorText != null)
            {
                currentActorText.text = $"현재 행동: {actor.GetName()}";
            }
            
            if (turnInfoText != null)
            {
                turnInfoText.text = $"턴 {battleSystem.currentTurn}";
            }
            
            if (turnProgressSlider != null)
            {
                float progress = (float)battleSystem.currentActorIndex / battleSystem.turnOrder.Count;
                turnProgressSlider.value = progress;
            }
        }
    }
    
    private void UpdateMechStatusUIs()
    {
        foreach (MechStatusUI statusUI in mechStatusUIs)
        {
            if (statusUI != null)
            {
                statusUI.UpdateStatus();
            }
        }
    }
    
    private void UpdateActionButtons()
    {
        if (currentMech == null) return;
        
        // AP에 따른 버튼 활성화/비활성화
        bool canAct = currentMech.CanAct();
        
        if (attackButton != null) attackButton.interactable = canAct && currentMech.actionPoints.currentAP >= 1;
        if (defendButton != null) defendButton.interactable = canAct && currentMech.actionPoints.currentAP >= 1;
        if (skillButton != null) skillButton.interactable = canAct;
        if (moveButton != null) moveButton.interactable = canAct && currentMech.actionPoints.currentAP >= 1;  // 이동 버튼 추가
        if (cooperationButton != null) cooperationButton.interactable = canAct;
        if (endTurnButton != null) endTurnButton.interactable = true;
    }
    
    private void UpdateAPDisplay()
    {
        if (currentMech == null) 
        {
            Debug.LogWarning("UpdateAPDisplay: currentMech가 null입니다!");
            return;
        }
        
        if (currentMech.actionPoints == null)
        {
            Debug.LogWarning($"UpdateAPDisplay: {currentMech.mechName}의 actionPoints가 null입니다!");
            return;
        }
        
        if (apText != null)
        {
            apText.text = $"AP: {currentMech.actionPoints.currentAP}/{currentMech.actionPoints.maxAP}";
        }
        else
        {
            Debug.LogWarning("UpdateAPDisplay: apText UI 요소가 없습니다!");
        }
        
        if (apSlider != null)
        {
            apSlider.value = currentMech.actionPoints.GetAPRatio();
        }
        else
        {
            Debug.LogWarning("UpdateAPDisplay: apSlider UI 요소가 없습니다!");
        }
    }
    
    private void OnAttackClicked()
    {
        if (currentMech == null) 
        {
            Debug.LogWarning("OnAttackClicked: 현재 기계가 설정되지 않았습니다!");
            Debug.Log("테스트용 기계를 자동 생성합니다...");
            CreateTestMech();
            if (currentMech == null)
            {
                Debug.LogError("기계 생성에 실패했습니다!");
                return;
            }
        }
        
        if (actionModeManager != null)
        {
            actionModeManager.SetActionMode(ActionModeManager.ActionMode.Attack, currentMech);
        }
        else
        {
            Debug.LogError("ActionModeManager를 찾을 수 없습니다!");
        }
    }
    
    private void OnDefendClicked()
    {
        if (currentMech == null) 
        {
            Debug.LogWarning("OnDefendClicked: 현재 기계가 설정되지 않았습니다!");
            Debug.Log("테스트용 기계를 자동 생성합니다...");
            CreateTestMech();
            if (currentMech == null)
            {
                Debug.LogError("기계 생성에 실패했습니다!");
                return;
            }
        }
        
        if (actionModeManager != null)
        {
            actionModeManager.SetActionMode(ActionModeManager.ActionMode.Defend, currentMech);
        }
        else
        {
            Debug.LogError("ActionModeManager를 찾을 수 없습니다!");
        }
        
        UpdateActionButtons();
        UpdateAPDisplay();
    }
    
    private void OnSkillClicked()
    {
        if (currentMech == null) 
        {
            Debug.LogWarning("OnSkillClicked: 현재 기계가 설정되지 않았습니다!");
            Debug.Log("테스트용 기계를 자동 생성합니다...");
            CreateTestMech();
            if (currentMech == null)
            {
                Debug.LogError("기계 생성에 실패했습니다!");
                return;
            }
        }
        
        if (actionModeManager != null)
        {
            actionModeManager.SetActionMode(ActionModeManager.ActionMode.Skill, currentMech);
        }
        else
        {
            Debug.LogError("ActionModeManager를 찾을 수 없습니다!");
        }
    }
    
    private void OnMoveClicked()
    {
        if (currentMech == null) 
        {
            Debug.LogWarning("OnMoveClicked: 현재 기계가 설정되지 않았습니다!");
            Debug.Log("테스트용 기계를 자동 생성합니다...");
            CreateTestMech();
            if (currentMech == null)
            {
                Debug.LogError("기계 생성에 실패했습니다!");
                return;
            }
        }
        
        if (actionModeManager != null)
        {
            actionModeManager.SetActionMode(ActionModeManager.ActionMode.Move, currentMech);
        }
        else
        {
            Debug.LogError("ActionModeManager를 찾을 수 없습니다!");
        }
    }
    
    private void OnCooperationClicked()
    {
        if (currentMech == null) return;
        
        ShowCooperationOptions();
    }
    
    private void OnRetreatClicked()
    {
        if (battleSystem != null)
        {
            battleSystem.AttemptRetreat();
        }
    }
    
    private void OnEndTurnClicked()
    {
        if (battleSystem != null)
        {
            battleSystem.EndCurrentTurn();
        }
    }
    
    private void ShowTargetSelection(string message, System.Action<MechCharacter> onTargetSelected)
    {
        // 타겟 선택 UI 구현
        Debug.Log(message);
        
        if (currentMech == null)
        {
            Debug.LogWarning("공격할 기계가 없습니다!");
            ShowDialogue("시스템", "공격할 기계가 없습니다!");
            return;
        }
        
        // 살아있는 적들 찾기
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        EnemyAI nearestEnemy = null;
        float shortestDistance = float.MaxValue;
        
        int aliveEnemyCount = 0;
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive)
            {
                aliveEnemyCount++;
                float distance = Vector3.Distance(currentMech.transform.position, enemy.transform.position);
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    nearestEnemy = enemy;
                }
            }
        }
        
        if (nearestEnemy != null && aliveEnemyCount > 0)
        {
            // AP 체크
            if (currentMech.actionPoints.currentAP < 1)
            {
                Debug.LogWarning($"{currentMech.mechName}의 AP가 부족합니다!");
                ShowDialogue(currentMech.mechName, "AP가 부족해서 공격할 수 없다!");
                return;
            }
            
            // 공격 실행
            float damage = currentMech.stats.attack;
            string attackMessage = $"{currentMech.mechName}이(가) {nearestEnemy.enemyName}을(를) {damage} 데미지로 공격!";
            
            Debug.Log(attackMessage);
            ShowDialogue(currentMech.mechName, "공격한다!");
            
            nearestEnemy.TakeDamage(damage);
            currentMech.ConsumeAP(1);
            
            UpdateActionButtons();
            UpdateAPDisplay();
            UpdateMechStatusUIs();
        }
        else
        {
            Debug.LogWarning("공격할 적이 없습니다!");
            ShowDialogue("시스템", "공격할 적이 없습니다!");
        }
    }
    
    private void OnAttackTargetSelected(MechCharacter target)
    {
        // 공격 대상이 선택되었을 때의 처리
        Debug.Log($"{currentMech.mechName}이 {target.mechName}을 공격합니다!");
    }
    
    private void ShowSkillSelection()
    {
        // 스킬 선택 UI 구현
        Debug.Log("스킬을 선택하세요");
        
        // 임시로 기본 스킬 사용
        if (currentMech.mechType == MechType.Rex)
        {
            RexMech rex = currentMech as RexMech;
            if (rex != null)
            {
                // 기존 ProtectiveStance 호출 대신 가디언 실드 사용으로 대체
                rex.UseGuardianShield();
            }
        }
        else if (currentMech.mechType == MechType.Luna)
        {
            LunaMech luna = currentMech as LunaMech;
            if (luna != null)
            {
                // 가장 손상된 아군을 찾아서 수리
                MechCharacter[] allies = FindObjectsOfType<MechCharacter>();
                MechCharacter mostDamaged = null;
                int lowestHP = int.MaxValue;
                
                foreach (MechCharacter ally in allies)
                {
                    if (ally != currentMech && ally.isAlive && ally.stats.currentHP < lowestHP)
                    {
                        lowestHP = ally.stats.currentHP;
                        mostDamaged = ally;
                    }
                }
                
                if (mostDamaged != null)
                {
                    // 기존 NanoRepair 호출 대신 새 메서드명 사용
                    luna.UseNanoRepair(mostDamaged, BodyPartType.Torso);
                }
            }
        }
        
        UpdateActionButtons();
        UpdateAPDisplay();
    }
    
    private void ShowCooperationOptions()
    {
        if (cooperationPanel == null) return;
        
        cooperationPanel.SetActive(true);
        
        // 기존 협력 옵션 제거
        foreach (Transform child in cooperationContainer)
        {
            Destroy(child.gameObject);
        }
        
        // 협력 가능한 아군들 찾기
        List<MechCharacter> allies = new List<MechCharacter>();
        MechCharacter[] allMechs = FindObjectsOfType<MechCharacter>();
        
        foreach (MechCharacter mech in allMechs)
        {
            if (mech != currentMech && mech.isAlive)
            {
                float distance = Vector3.Distance(currentMech.transform.position, mech.transform.position);
                if (distance <= 3f) // 협력 가능 거리
                {
                    allies.Add(mech);
                }
            }
        }
        
        // 협력 옵션 생성
        foreach (MechCharacter ally in allies)
        {
            if (cooperationOptionPrefab != null)
            {
                GameObject option = Instantiate(cooperationOptionPrefab, cooperationContainer);
                CooperationOptionUI optionUI = option.GetComponent<CooperationOptionUI>();
                if (optionUI != null)
                {
                    optionUI.Initialize(currentMech, ally, this);
                }
            }
        }
    }
    
    public void HideCooperationPanel()
    {
        if (cooperationPanel != null)
        {
            cooperationPanel.SetActive(false);
        }
    }
    
    public void ShowDialogue(string speaker, string dialogue)
    {
        if (dialoguePanel != null)
        {
            dialoguePanel.SetActive(true);
            
            if (speakerName != null)
            {
                speakerName.text = speaker;
            }
            
            if (dialogueText != null)
            {
                dialogueText.text = dialogue;
            }
        }
        
        // 3초 후 대화창 숨기기
        Invoke("HideDialogue", 3f);
    }
    
    private void HideDialogue()
    {
        if (dialoguePanel != null)
        {
            dialoguePanel.SetActive(false);
        }
    }
    
    // 테스트용 메서드들
    [ContextMenu("UI 테스트 - 공격 버튼")]
    public void TestAttackButton()
    {
        Debug.Log("공격 버튼 테스트 - OnAttackClicked() 호출");
        OnAttackClicked();
    }
    
    [ContextMenu("UI 테스트 - 방어 버튼")]
    public void TestDefendButton()
    {
        Debug.Log("방어 버튼 테스트 - OnDefendClicked() 호출");
        OnDefendClicked();
    }
    
    [ContextMenu("UI 테스트 - 대화창 표시")]
    public void TestShowDialogue()
    {
        ShowDialogue("테스트", "이것은 테스트 대화입니다!");
    }
    
    [ContextMenu("UI 테스트 - 모든 패널 활성화")]
    public void TestActivateAllPanels()
    {
        if (battlePanel != null) 
        {
            battlePanel.SetActive(true);
            Debug.Log("BattlePanel 활성화됨");
        }
        if (actionPanel != null) 
        {
            actionPanel.SetActive(true);
            Debug.Log("ActionPanel 활성화됨");
        }
        if (statusPanel != null) 
        {
            statusPanel.SetActive(true);
            Debug.Log("StatusPanel 활성화됨");
        }
        if (dialoguePanel != null) 
        {
            dialoguePanel.SetActive(true);
            Debug.Log("DialoguePanel 활성화됨");
        }
    }
    
    [ContextMenu("UI 테스트 - 테스트용 기계 생성")]
    public void CreateTestMech()
    {
        if (currentMech == null)
        {
            // 테스트용 기계 생성
            GameObject testMechObj = new GameObject("TestMech");
            currentMech = testMechObj.AddComponent<RexMech>();
            
            // 아군 위치 (아래쪽)에 기계 배치
            PositionMechInPlayerArea(testMechObj);
            
            // 기본 설정
            currentMech.mechName = "테스트 렉스";
            currentMech.mechType = MechType.Rex;
            currentMech.isAlive = true;
            
            // 스탯 설정
            currentMech.stats = new MechStats
            {
                maxHP = 100,
                currentHP = 80,
                attack = 25,
                defense = 15
            };
            
            // AP 설정
            currentMech.actionPoints = new ActionPoint
            {
                maxAP = 3,
                currentAP = 2
            };
            
            Debug.Log($"테스트용 기계 생성: {currentMech.mechName} (HP: {currentMech.stats.currentHP}/{currentMech.stats.maxHP}, AP: {currentMech.actionPoints.currentAP}/{currentMech.actionPoints.maxAP})");
            
            // UI 업데이트
            UpdateActionButtons();
            UpdateAPDisplay();
        }
        else
        {
            Debug.Log($"현재 기계가 이미 설정됨: {currentMech.mechName}");
            
            // 기존 기계가 잘못된 위치에 있는지 확인
            Vector3 currentPos = currentMech.transform.position;
            if (Vector3.Distance(currentPos, Vector3.zero) < 0.5f)
            {
                Debug.Log("기존 기계가 중앙에 있어서 아군 지역으로 이동시킵니다.");
                PositionMechInPlayerArea(currentMech.gameObject);
            }
        }
    }
    
    /// <summary>
    /// 기계를 아군 지역(격자 하단)에 배치합니다
    /// </summary>
    private void PositionMechInPlayerArea(GameObject mechObj)
    {
        BattleGridManager gridManager = FindObjectOfType<BattleGridManager>();
        if (gridManager == null)
        {
            Debug.LogWarning("BattleGridManager를 찾을 수 없어서 기본 위치에 배치합니다.");
            mechObj.transform.position = new Vector3(0, -3, 0); // 기본 아래쪽 위치
            return;
        }
        
        // 아군 지역 격자 좌표 (격자 하단 중앙부)
        int playerGridX = 5;  // 격자 중앙 x좌표 (10x10 격자의 경우)
        int playerGridY = 1;  // 격자 하단에서 약간 위
        
        // 격자 좌표를 월드 좌표로 변환
        Vector3 worldPos = gridManager.GetWorldPosition(playerGridX, playerGridY);
        Vector3 centerPos = worldPos + new Vector3(gridManager.gridSize / 2f, gridManager.gridSize / 2f, 0);
        
        // 기계를 해당 위치에 배치
        mechObj.transform.position = centerPos;
        
        // 시각적 표현 추가 (파란색 쿼드)
        CreateMechVisualRepresentation(mechObj, gridManager.gridSize);
        
        Debug.Log($"테스트 기계를 아군 지역에 배치: 격자({playerGridX}, {playerGridY}) -> 월드({centerPos})");
    }
    
    /// <summary>
    /// 기계의 시각적 표현을 생성합니다
    /// </summary>
    private void CreateMechVisualRepresentation(GameObject mechObj, float gridSize)
    {
        // 기존 시각적 표현 제거
        Transform existing = mechObj.transform.Find("VisualQuad");
        if (existing != null)
        {
            DestroyImmediate(existing.gameObject);
        }
        
        // 2D용 시각적 표현을 위한 쿼드 생성
        GameObject visualQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        visualQuad.name = "VisualQuad";
        visualQuad.transform.parent = mechObj.transform;
        visualQuad.transform.localPosition = Vector3.zero;
        visualQuad.transform.localScale = new Vector3(gridSize * 0.9f, gridSize * 0.9f, 1);
        
        // 파란색 머티리얼 적용 (아군)
        Renderer renderer = visualQuad.GetComponent<Renderer>();
        Material blueMaterial = new Material(Shader.Find("Sprites/Default"));
        blueMaterial.color = Color.blue;
        blueMaterial.renderQueue = 2000; // 격자보다 앞에 렌더링
        renderer.material = blueMaterial;
        
        // 콜라이더 제거 (시각적 표현만 필요)
        Collider quadCollider = visualQuad.GetComponent<Collider>();
        if (quadCollider != null)
        {
            DestroyImmediate(quadCollider);
        }
    }
    
    /// <summary>
    /// 현재 기계를 아군 지역으로 재배치합니다
    /// </summary>
    [ContextMenu("기계 위치 재설정")]
    public void ResetMechPosition()
    {
        if (currentMech != null)
        {
            Debug.Log("기계를 아군 지역으로 재배치합니다.");
            PositionMechInPlayerArea(currentMech.gameObject);
        }
        else
        {
            Debug.LogWarning("재배치할 기계가 없습니다!");
        }
    }
}

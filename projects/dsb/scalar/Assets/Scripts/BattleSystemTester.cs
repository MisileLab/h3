using UnityEngine;

/// <summary>
/// 새로운 전투 시스템을 테스트하기 위한 스크립트
/// </summary>
public class BattleSystemTester : MonoBehaviour
{
    [Header("테스트 설정")]
    public bool enableTesting = true;
    public KeyCode testKey = KeyCode.T;
    public KeyCode createMechKey = KeyCode.M;      // M 키로 기계 생성
    public KeyCode startBattleKey = KeyCode.B;    // B 키로 전투 시작
    
    private ActionModeManager actionModeManager;
    private RangeDisplay rangeDisplay;
    private TargetSelector targetSelector;
    private BattleUI battleUI;
    
    private void Start()
    {
        if (!enableTesting) return;
        
        // 시스템 참조 가져오기
        actionModeManager = FindObjectOfType<ActionModeManager>();
        rangeDisplay = FindObjectOfType<RangeDisplay>();
        targetSelector = FindObjectOfType<TargetSelector>();
        battleUI = FindObjectOfType<BattleUI>();
        
        Debug.Log("=== 새로운 전투 시스템 테스터 활성화됨 ===");
        Debug.Log("🔥 단축키 모음:");
        Debug.Log("T 키 - 시스템 테스트");
        Debug.Log("M 키 - 테스트용 기계 생성"); 
        Debug.Log("B 키 - 전체 전투 시스템 초기화");
        Debug.Log("ESC 키 - 선택 취소");
        Debug.Log("공격/이동/스킬/방어 버튼 클릭 후 격자를 클릭하세요.");
    }
    
    private void Update()
    {
        if (!enableTesting) return;
        
        if (Input.GetKeyDown(testKey))
        {
            RunSystemTest();
        }
        
        if (Input.GetKeyDown(createMechKey))
        {
            CreateTestMech();
        }
        
        if (Input.GetKeyDown(startBattleKey))
        {
            InitializeBattleSystem();
        }
    }
    
    private void RunSystemTest()
    {
        Debug.Log("=== 시스템 테스트 시작 ===");
        
        // 1. 시스템 컴포넌트 체크
        TestSystemComponents();
        
        // 2. UI 체크
        TestUIComponents();
        
        // 3. 범위 표시 테스트
        TestRangeDisplay();
        
        Debug.Log("=== 시스템 테스트 완료 ===");
        Debug.Log("이제 UI 버튼들을 클릭해서 실제 기능을 테스트해보세요!");
    }
    
    private void TestSystemComponents()
    {
        Debug.Log("--- 시스템 컴포넌트 체크 ---");
        
        if (actionModeManager != null)
        {
            Debug.Log("✅ ActionModeManager 발견됨");
        }
        else
        {
            Debug.LogError("❌ ActionModeManager 없음!");
        }
        
        if (rangeDisplay != null)
        {
            Debug.Log("✅ RangeDisplay 발견됨");
        }
        else
        {
            Debug.LogError("❌ RangeDisplay 없음!");
        }
        
        if (targetSelector != null)
        {
            Debug.Log("✅ TargetSelector 발견됨");
        }
        else
        {
            Debug.LogError("❌ TargetSelector 없음!");
        }
        
        BattleGridManager gridManager = FindObjectOfType<BattleGridManager>();
        if (gridManager != null)
        {
            Debug.Log("✅ BattleGridManager 발견됨");
        }
        else
        {
            Debug.LogError("❌ BattleGridManager 없음!");
        }
    }
    
    private void TestUIComponents()
    {
        Debug.Log("--- UI 컴포넌트 체크 ---");
        
        if (battleUI != null)
        {
            Debug.Log("✅ BattleUI 발견됨");
            
            // 버튼들 체크
            if (battleUI.attackButton != null) Debug.Log("✅ 공격 버튼 OK");
            else Debug.LogWarning("⚠️ 공격 버튼 없음");
            
            if (battleUI.defendButton != null) Debug.Log("✅ 방어 버튼 OK");
            else Debug.LogWarning("⚠️ 방어 버튼 없음");
            
            if (battleUI.skillButton != null) Debug.Log("✅ 스킬 버튼 OK");
            else Debug.LogWarning("⚠️ 스킬 버튼 없음");
            
            if (battleUI.moveButton != null) Debug.Log("✅ 이동 버튼 OK");
            else Debug.LogWarning("⚠️ 이동 버튼 없음");
        }
        else
        {
            Debug.LogError("❌ BattleUI 없음!");
        }
    }
    
    private void TestRangeDisplay()
    {
        if (rangeDisplay == null) return;
        
        Debug.Log("--- 범위 표시 테스트 ---");
        
        // 2초 후 테스트 범위 표시
        Invoke("ShowTestRanges", 2f);
    }
    
    private void ShowTestRanges()
    {
        if (rangeDisplay == null) return;
        
        Vector3 testCenter = Vector3.zero;
        
        Debug.Log("테스트: 공격 범위 표시 (2초간)");
        rangeDisplay.ShowAttackRange(testCenter, 2);
        
        // 3초 후 이동 범위로 변경
        Invoke("ShowTestMoveRange", 3f);
    }
    
    private void ShowTestMoveRange()
    {
        if (rangeDisplay == null) return;
        
        Vector3 testCenter = Vector3.zero;
        
        Debug.Log("테스트: 이동 범위 표시 (2초간)");
        rangeDisplay.ShowMoveRange(testCenter, 3);
        
        // 3초 후 정리
        Invoke("ClearTestRanges", 3f);
    }
    
    private void ClearTestRanges()
    {
        if (rangeDisplay == null) return;
        
        Debug.Log("테스트 범위 표시 정리");
        rangeDisplay.ClearAll();
    }
    
    /// <summary>
    /// 콘솔에 도움말을 출력합니다
    /// </summary>
    [ContextMenu("도움말 표시")]
    public void ShowHelp()
    {
        Debug.Log("=== 새로운 전투 시스템 사용법 ===");
        Debug.Log("1. 공격 버튼 클릭 → 빨간색 사정거리 표시 → 적 클릭하여 공격");
        Debug.Log("2. 이동 버튼 클릭 → 파란색 이동 범위 표시 → 빈 곳 클릭하여 이동");
        Debug.Log("3. 스킬 버튼 클릭 → 초록색 스킬 범위 표시 → 대상 클릭하여 스킬 사용");
        Debug.Log("4. 방어 버튼 클릭 → 즉시 방어 실행 (타겟 선택 불필요)");
        Debug.Log("5. ESC 키로 언제든 선택 취소 가능");
        Debug.Log("6. T 키로 시스템 테스트 실행");
        Debug.Log("=======================================");
    }
    
    /// <summary>
    /// 테스트용 기계를 생성합니다
    /// </summary>
    private void CreateTestMech()
    {
        Debug.Log("=== 테스트용 기계 생성 시작 ===");
        
        if (battleUI != null)
        {
            // BattleUI의 테스트 기계 생성 기능 사용
            battleUI.CreateTestMech();
            Debug.Log("✅ 테스트용 기계 생성 완료");
            
            // 생성된 기계 위치 확인
            var currentMech = battleUI.GetType().GetField("currentMech", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(battleUI) as MechCharacter;
            if (currentMech != null)
            {
                Debug.Log($"생성된 기계 위치: {currentMech.transform.position}");
            }
        }
        else
        {
            Debug.LogError("❌ BattleUI를 찾을 수 없어 기계를 생성할 수 없습니다!");
        }
    }
    
    /// <summary>
    /// 전체 전투 시스템을 초기화합니다
    /// </summary>
    private void InitializeBattleSystem()
    {
        Debug.Log("=== 전체 전투 시스템 초기화 시작 ===");
        
        // BattleInitializer 찾기
        BattleInitializer battleInitializer = FindObjectOfType<BattleInitializer>();
        
        if (battleInitializer != null)
        {
            // 전투 시스템 초기화 및 시작
            battleInitializer.InitializeAndStartBattle();
            Debug.Log("✅ 전투 시스템 초기화 완료");
        }
        else
        {
            Debug.LogWarning("BattleInitializer가 없어서 수동으로 초기화합니다...");
            
            // 수동으로 기본 시스템 생성
            CreateBasicBattleSystem();
        }
    }
    
    /// <summary>
    /// 기본 전투 시스템을 수동으로 생성합니다
    /// </summary>
    private void CreateBasicBattleSystem()
    {
        Debug.Log("--- 기본 전투 시스템 수동 생성 ---");
        
        // 격자 시스템 생성
        CreateGridSystemIfNeeded();
        
        // UI 시스템 생성
        CreateUISystemIfNeeded();
        
        // 테스트용 기계 생성
        CreateTestMech();
        
        Debug.Log("✅ 기본 전투 시스템 생성 완료");
    }
    
    private void CreateGridSystemIfNeeded()
    {
        BattleGridManager gridManager = FindObjectOfType<BattleGridManager>();
        if (gridManager == null)
        {
            GameObject gridObj = new GameObject("BattleGridManager");
            gridManager = gridObj.AddComponent<BattleGridManager>();
            gridManager.showGrid = true;
            gridManager.gridSize = 1f;
            gridManager.gridWidth = 10;
            gridManager.gridHeight = 10;
            gridManager.CreateGrid();
            Debug.Log("✅ 격자 시스템 생성됨");
        }
        else
        {
            Debug.Log("✅ 격자 시스템 이미 존재");
        }
    }
    
    private void CreateUISystemIfNeeded()
    {
        if (battleUI == null)
        {
            // UIAutoGenerator를 사용해서 UI 생성
            UIAutoGenerator uiGenerator = FindObjectOfType<UIAutoGenerator>();
            if (uiGenerator == null)
            {
                GameObject uiGenObj = new GameObject("UIAutoGenerator");
                uiGenerator = uiGenObj.AddComponent<UIAutoGenerator>();
            }
            
            battleUI = uiGenerator.CreateBattleUI();
            Debug.Log("✅ UI 시스템 생성됨");
        }
        else
        {
            Debug.Log("✅ UI 시스템 이미 존재");
        }
    }
}

using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class BattleInitializer : MonoBehaviour
{
    [Header("테스트 설정")]
    public bool autoStartBattle = false;
    public float battleStartDelay = 2f;
    
    [Header("테스트 기계들")]
    public GameObject rexPrefab;
    public GameObject lunaPrefab;
    public GameObject zeroPrefab;
    public GameObject novaPrefab;
    
    [Header("테스트 적들")]
    public GameObject enemyPrefab;
    public int enemyCount = 2;  // 적 스폰 포인트와 동일하게 설정
    
    [Header("스폰 위치")]
    public Transform[] playerSpawnPoints;
    public Transform[] enemySpawnPoints;
    
    [Header("자동 생성 설정")]
    public bool createDummyDataIfEmpty = true;
    public bool createGridSystem = true;
    public bool createUIAutomatically = true;
    
    [System.Serializable]
    public struct GridPosition
    {
        public int x;
        public int y;
        public GridPosition(int x, int y) { this.x = x; this.y = y; }
    }
    
    [Header("격자 기반 스폰 위치")]
    public GridPosition[] defaultPlayerGridPositions = {
        new GridPosition(1, 0),  // 렉스
        new GridPosition(2, 0),  // 루나  
        new GridPosition(3, 0),  // 제로
        new GridPosition(4, 0)   // 노바
    };
    public GridPosition[] defaultEnemyGridPositions = {
        new GridPosition(8, 0),  // 적1
        new GridPosition(9, 0)   // 적2
    };
    
    [Header("UI 자동 생성 설정")]
    public Font defaultFont;
    
    // UI 생성 시 사용할 private 필드들
    private UIAutoGenerator uiGenerator;
    private BattleUI battleUI;
    
    private void Start()
    {
        if (autoStartBattle)
        {
            Invoke("InitializeAndStartBattle", battleStartDelay);
        }
    }
    
    /// <summary>
    /// 격자 좌표를 월드 중앙 위치로 변환하는 헬퍼 메서드
    /// BattleGridManager를 참조해서 정확한 좌표 계산
    /// </summary>
    /// <param name="gridX">격자 X 좌표 (0~9)</param>
    /// <param name="gridY">격자 Y 좌표 (0~9)</param>
    /// <returns>격자 칸의 중앙 월드 좌표</returns>
    public Vector3 GetGridCenterWorldPosition(int gridX, int gridY)
    {
        // BattleGridManager를 찾아서 정확한 좌표 계산
        BattleGridManager gridManager = FindObjectOfType<BattleGridManager>();
        if (gridManager != null)
        {
            Vector3 gridCorner = gridManager.GetWorldPosition(gridX, gridY);
            return gridCorner + new Vector3(0.5f, 0.5f, 0f); // 격자 칸의 중앙으로 이동
        }
        
        // 폴백: 기본 계산
        Vector3 gridStartPos = new Vector3(-5f, -5f, 0f);
        return gridStartPos + new Vector3(gridX + 0.5f, gridY + 0.5f, 0f);
    }
    
    /// <summary>
    /// 기존에 생성된 오브젝트들을 정리합니다 (가운데 적 포함)
    /// </summary>
    private void ClearExistingObjects()
    {
        Debug.Log("=== 기존 오브젝트 정리 시작 ===");
        
        // 모든 GameObject 검사해서 의심스러운 것들 제거
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        int removedCount = 0;
        
        foreach (GameObject obj in allObjects)
        {
            // null이거나 시스템 오브젝트는 건드리지 않음
            if (obj == null || obj == this.gameObject) continue;
            
            // 제거 대상 체크
            bool shouldRemove = false;
            string removeReason = "";
            
            // 1. EnemyAI 컴포넌트가 있는 경우
            if (obj.GetComponent<EnemyAI>() != null)
            {
                shouldRemove = true;
                removeReason = "EnemyAI 컴포넌트";
            }
            // 2. 기계 관련 컴포넌트가 있는 경우  
            else if (obj.GetComponent<RexMech>() != null || obj.GetComponent<LunaMech>() != null || 
                     obj.GetComponent<ZeroMech>() != null || obj.GetComponent<NovaMech>() != null)
            {
                shouldRemove = true;
                removeReason = "기계 컴포넌트";
            }
            // 3. 이름 패턴으로 확인
            else if (obj.name.Contains("Rex") || obj.name.Contains("Luna") || 
                     obj.name.Contains("Zero") || obj.name.Contains("Nova") ||
                     obj.name.Contains("Dummy") || obj.name.Contains("Enemy") ||
                     obj.name.Contains("Test") || obj.name.Contains("Spawn"))
            {
                shouldRemove = true;
                removeReason = $"이름 패턴: {obj.name}";
            }
            // 4. 부모가 없는 Quad나 Cube (격자 제외)
            else if (obj.transform.parent == null && 
                     (obj.name.Contains("Quad") || obj.name.Contains("Cube")) &&
                     !obj.name.Contains("Grid") && !obj.name.Contains("Background"))
            {
                shouldRemove = true;
                removeReason = $"고아 프리미티브: {obj.name}";
            }
            // 5. 렌더러가 있고 빨간색이나 파란색인 경우 (격자 제외)
            else if (obj.GetComponent<Renderer>() != null && !obj.name.Contains("Grid"))
            {
                Renderer renderer = obj.GetComponent<Renderer>();
                if (renderer.material != null)
                {
                    Color color = renderer.material.color;
                    if ((color == Color.red || color == Color.blue) && obj.transform.parent == null)
                    {
                        shouldRemove = true;
                        removeReason = $"색상 매치: {color}";
                    }
                }
            }
            // 6. 가운데 위치(0,0,0 근처)에 있는 모든 렌더러 오브젝트 (격자 제외)
            else if (obj.GetComponent<Renderer>() != null && !obj.name.Contains("Grid") && !obj.name.Contains("Background"))
            {
                Vector3 pos = obj.transform.position;
                if (Vector3.Distance(pos, Vector3.zero) < 1f) // 중앙에서 1 유닛 이내
                {
                    shouldRemove = true;
                    removeReason = $"중앙 위치: {pos}";
                }
            }
            
            if (shouldRemove)
            {
                Debug.Log($"오브젝트 제거: {obj.name} (이유: {removeReason})");
                DestroyImmediate(obj);
                removedCount++;
            }
        }
        
        Debug.Log($"=== 기존 오브젝트 정리 완료: {removedCount}개 제거 ===");
    }
    
    public void InitializeAndStartBattle()
    {
        // 0. 기존 오브젝트들 정리
        ClearExistingObjects();
        
        // 1. 격자 시스템 먼저 생성
        if (createGridSystem)
        {
            CreateGridSystem();
        }
        
        // 2. 더미 데이터 자동 생성 (프리팹만)
        if (createDummyDataIfEmpty)
        {
            CreateDummyDataIfNeeded();
        }
        
        // 3. 격자 시스템이 생성된 후 스폰 포인트 생성
        CreateDefaultSpawnPoints();
        
        // 4. 플레이어 기계들 생성
        CreatePlayerMechs();
        
        // 5. 적들 생성
        CreateEnemies();
        
        // 6. UI 자동 생성
        if (createUIAutomatically)
        {
            CreateBattleUI();
        }
        
        // 7. 전투 시작
        StartBattle();
    }
    
    private void CreateDummyDataIfNeeded()
    {
        // 프리팹이 없으면 더미 기계 생성
        if (rexPrefab == null) rexPrefab = CreateDummyMech("Rex", MechType.Rex);
        if (lunaPrefab == null) lunaPrefab = CreateDummyMech("Luna", MechType.Luna);
        if (zeroPrefab == null) zeroPrefab = CreateDummyMech("Zero", MechType.Zero);
        if (novaPrefab == null) novaPrefab = CreateDummyMech("Nova", MechType.Nova);
        
        // 적 프리팹이 없으면 더미 적 생성
        if (enemyPrefab == null) enemyPrefab = CreateDummyEnemy();
    }
    
    private GameObject CreateDummyMech(string name, MechType mechType)
    {
        GameObject dummyMech = new GameObject(name);
        dummyMech.transform.position = new Vector3(-1000, -1000, 0); // 화면 밖으로 임시 이동
        
        // 2D용 시각적 표현을 위한 쿼드 생성
        GameObject visualQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        visualQuad.name = "VisualQuad";
        visualQuad.transform.parent = dummyMech.transform;
        visualQuad.transform.localPosition = Vector3.zero;
        visualQuad.transform.localScale = new Vector3(0.9f, 0.9f, 1); // 격자 칸에 딱 맞게 (약간 여백)
        
        // 파란색 머티리얼 적용 (아군) - 캐릭터가 앞에 보이도록 설정
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
        
        // 기계 타입에 따른 스크립트 추가
        switch (mechType)
        {
            case MechType.Rex:
                dummyMech.AddComponent<RexMech>();
                break;
            case MechType.Luna:
                dummyMech.AddComponent<LunaMech>();
                break;
            case MechType.Zero:
                dummyMech.AddComponent<ZeroMech>();
                break;
            case MechType.Nova:
                dummyMech.AddComponent<NovaMech>();
                break;
        }
        
        Debug.Log($"더미 기계 생성: {name}");
        return dummyMech;
    }
    
    private GameObject CreateDummyEnemy()
    {
        GameObject dummyEnemy = new GameObject("DummyEnemy");
        dummyEnemy.transform.position = new Vector3(-1000, -1000, 0); // 화면 밖으로 임시 이동
        
        // 2D용 시각적 표현을 위한 쿼드 생성
        GameObject visualQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        visualQuad.name = "VisualQuad";
        visualQuad.transform.parent = dummyEnemy.transform;
        visualQuad.transform.localPosition = Vector3.zero;
        visualQuad.transform.localScale = new Vector3(0.9f, 0.9f, 1); // 격자 칸에 딱 맞게 (약간 여백)
        
        // 빨간색 머티리얼 적용 (적) - 캐릭터가 앞에 보이도록 설정
        Renderer renderer = visualQuad.GetComponent<Renderer>();
        Material redMaterial = new Material(Shader.Find("Sprites/Default"));
        redMaterial.color = Color.red;
        redMaterial.renderQueue = 2000; // 격자보다 앞에 렌더링
        renderer.material = redMaterial;
        
        // 콜라이더 제거 (시각적 표현만 필요)
        Collider quadCollider = visualQuad.GetComponent<Collider>();
        if (quadCollider != null)
        {
            DestroyImmediate(quadCollider);
        }
        
        // EnemyAI 컴포넌트 추가 및 설정
        EnemyAI enemyAI = dummyEnemy.AddComponent<EnemyAI>();
        enemyAI.enemyName = "더미 적";
        enemyAI.enemyType = EnemyType.Scrapper;
        enemyAI.maxHP = 80;
        enemyAI.currentHP = 80;
        enemyAI.attack = 20;
        enemyAI.defense = 10;
        enemyAI.speed = 8;
        
        Debug.Log("더미 적 생성: DummyEnemy");
        return dummyEnemy;
    }
    
    private void CreateDefaultSpawnPoints()
    {
        // 플레이어 스폰 포인트 생성
        GameObject playerSpawnParent = new GameObject("PlayerSpawnPoints");
        playerSpawnPoints = new Transform[defaultPlayerGridPositions.Length];
        
        for (int i = 0; i < defaultPlayerGridPositions.Length; i++)
        {
            GameObject spawnPoint = new GameObject($"PlayerSpawn_{i}");
            spawnPoint.transform.parent = playerSpawnParent.transform;
            Vector3 worldPos = GetGridCenterWorldPosition(defaultPlayerGridPositions[i].x, defaultPlayerGridPositions[i].y);
            spawnPoint.transform.position = worldPos;
            playerSpawnPoints[i] = spawnPoint.transform;
            Debug.Log($"플레이어 스폰 {i}: 격자({defaultPlayerGridPositions[i].x},{defaultPlayerGridPositions[i].y}) -> 월드{worldPos}");
        }
        
        // 적 스폰 포인트 생성
        GameObject enemySpawnParent = new GameObject("EnemySpawnPoints");
        enemySpawnPoints = new Transform[defaultEnemyGridPositions.Length];
        
        for (int i = 0; i < defaultEnemyGridPositions.Length; i++)
        {
            GameObject spawnPoint = new GameObject($"EnemySpawn_{i}");
            spawnPoint.transform.parent = enemySpawnParent.transform;
            Vector3 worldPos = GetGridCenterWorldPosition(defaultEnemyGridPositions[i].x, defaultEnemyGridPositions[i].y);
            spawnPoint.transform.position = worldPos;
            enemySpawnPoints[i] = spawnPoint.transform;
            Debug.Log($"적 스폰 {i}: 격자({defaultEnemyGridPositions[i].x},{defaultEnemyGridPositions[i].y}) -> 월드{worldPos}");
        }
        
        Debug.Log("기본 스폰 포인트 생성 완료");
    }
    
    private void CreateGridSystem()
    {
        // 이미 격자 시스템이 있으면 생성하지 않음
        BattleGridManager existingGrid = FindObjectOfType<BattleGridManager>();
        if (existingGrid != null)
        {
            Debug.Log("격자 시스템이 이미 존재합니다.");
            return;
        }
        
        // 격자 시스템 GameObject 생성
        GameObject gridSystem = new GameObject("BattleGridSystem");
        BattleGridManager gridManager = gridSystem.AddComponent<BattleGridManager>();
        
        // 격자 설정
        gridManager.showGrid = true;
        gridManager.gridSize = 1f;
        gridManager.gridWidth = 10;
        gridManager.gridHeight = 10;
        gridManager.gridColor = new Color(1f, 1f, 1f, 0.3f); // 반투명 흰색
        gridManager.gridLineWidth = 0.05f;
        gridManager.centerGrid = true;
        gridManager.gridCenter = Vector3.zero;
        
        // 격자 생성
        gridManager.CreateGrid();
        
        Debug.Log("격자 시스템 생성 완료");
    }
    
    private void CreatePlayerMechs()
    {
        // 렉스 생성
        if (rexPrefab != null && playerSpawnPoints != null && playerSpawnPoints.Length > 0)
        {
            GameObject rex = Instantiate(rexPrefab, playerSpawnPoints[0].position, playerSpawnPoints[0].rotation);
            RexMech rexMech = rex.GetComponent<RexMech>();
            if (rexMech == null)
            {
                rexMech = rex.AddComponent<RexMech>();
            }
            Debug.Log($"렉스 생성 완료 - 위치: {playerSpawnPoints[0].position}");
        }
        
        // 루나 생성
        if (lunaPrefab != null && playerSpawnPoints != null && playerSpawnPoints.Length > 1)
        {
            GameObject luna = Instantiate(lunaPrefab, playerSpawnPoints[1].position, playerSpawnPoints[1].rotation);
            LunaMech lunaMech = luna.GetComponent<LunaMech>();
            if (lunaMech == null)
            {
                lunaMech = luna.AddComponent<LunaMech>();
            }
            Debug.Log($"루나 생성 완료 - 위치: {playerSpawnPoints[1].position}");
        }
        
        // 제로 생성
        if (zeroPrefab != null && playerSpawnPoints != null && playerSpawnPoints.Length > 2)
        {
            GameObject zero = Instantiate(zeroPrefab, playerSpawnPoints[2].position, playerSpawnPoints[2].rotation);
            ZeroMech zeroMech = zero.GetComponent<ZeroMech>();
            if (zeroMech == null)
            {
                zeroMech = zero.AddComponent<ZeroMech>();
            }
            Debug.Log($"제로 생성 완료 - 위치: {playerSpawnPoints[2].position}");
        }
        
        // 노바 생성
        if (novaPrefab != null && playerSpawnPoints != null && playerSpawnPoints.Length > 3)
        {
            GameObject nova = Instantiate(novaPrefab, playerSpawnPoints[3].position, playerSpawnPoints[3].rotation);
            NovaMech novaMech = nova.GetComponent<NovaMech>();
            if (novaMech == null)
            {
                novaMech = nova.AddComponent<NovaMech>();
            }
            Debug.Log($"노바 생성 완료 - 위치: {playerSpawnPoints[3].position}");
        }
    }
    
    private void CreateEnemies()
    {
        // 적 생성 개수를 스폰 포인트 개수로 제한
        int actualEnemyCount = Mathf.Min(enemyCount, enemySpawnPoints != null ? enemySpawnPoints.Length : 0);
        
        for (int i = 0; i < actualEnemyCount; i++)
        {
            if (enemyPrefab != null && enemySpawnPoints != null && enemySpawnPoints.Length > i)
            {
                GameObject enemy = Instantiate(enemyPrefab, enemySpawnPoints[i].position, enemySpawnPoints[i].rotation);
                EnemyAI enemyAI = enemy.GetComponent<EnemyAI>();
                if (enemyAI == null)
                {
                    enemyAI = enemy.AddComponent<EnemyAI>();
                }
                
                // 적 설정
                enemyAI.enemyName = $"적 {i + 1}";
                enemyAI.enemyType = (EnemyType)(i % 4); // 4가지 타입 순환
                enemyAI.maxHP = 80 + i * 20;
                enemyAI.currentHP = enemyAI.maxHP;
                enemyAI.attack = 15 + i * 5;
                enemyAI.defense = 8 + i * 2;
                enemyAI.speed = 10 + i * 2;
                
                Debug.Log($"적 {i + 1} 생성 완료: {enemyAI.enemyName} - 위치: {enemySpawnPoints[i].position}");
            }
        }
        
        Debug.Log($"적 생성 완료 - 총 {actualEnemyCount}마리 (요청: {enemyCount}, 스폰 포인트: {(enemySpawnPoints != null ? enemySpawnPoints.Length : 0)})");
    }
    
    private void StartBattle()
    {
        GameManager gameManager = FindObjectOfType<GameManager>();
        if (gameManager != null)
        {
            gameManager.StartBattle();
        }
        else
        {
            BattleSystem battleSystem = FindObjectOfType<BattleSystem>();
            if (battleSystem != null)
            {
                battleSystem.StartBattle();
            }
        }
    }
    
    [ContextMenu("테스트 전투 시작")]
    public void TestStartBattle()
    {
        InitializeAndStartBattle();
    }
    
    [ContextMenu("기계들만 생성")]
    public void CreateMechsOnly()
    {
        CreatePlayerMechs();
        CreateEnemies();
    }
    
    [ContextMenu("UI만 생성")]
    public void CreateUIOnly()
    {
        CreateBattleUI();
    }
    
    [ContextMenu("중앙 오브젝트 제거")]
    public void ClearCenterObjects()
    {
        Debug.Log("=== 중앙 오브젝트 제거 시작 ===");
        
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        int removedCount = 0;
        
        foreach (GameObject obj in allObjects)
        {
            if (obj == null || obj == this.gameObject) continue;
            
            // 중앙 근처에 있는 렌더러 오브젝트 제거 (격자 및 시스템 오브젝트 제외)
            if (obj.GetComponent<Renderer>() != null && 
                !obj.name.Contains("Grid") && 
                !obj.name.Contains("Background") &&
                !obj.name.Contains("Camera") &&
                !obj.name.Contains("Light"))
            {
                Vector3 pos = obj.transform.position;
                if (Vector3.Distance(pos, Vector3.zero) < 2f) // 중앙에서 2 유닛 이내
                {
                    Debug.Log($"중앙 오브젝트 제거: {obj.name} (위치: {pos})");
                    DestroyImmediate(obj);
                    removedCount++;
                }
            }
        }
        
        Debug.Log($"=== 중앙 오브젝트 제거 완료: {removedCount}개 제거 ===");
    }
    
    /// <summary>
    /// 전투 UI를 자동으로 생성합니다
    /// </summary>
    private void CreateBattleUI()
    {
        Debug.Log("=== UI 자동 생성 시작 ===");
        
        // UI 자동 생성기 생성
        GameObject uiGeneratorObj = new GameObject("UIAutoGenerator");
        uiGenerator = uiGeneratorObj.AddComponent<UIAutoGenerator>();
        uiGenerator.defaultFont = defaultFont;
        
        // UI 자동 생성
        battleUI = uiGenerator.CreateBattleUI();
        
        Debug.Log("=== UI 자동 생성 완료 ===");
    }
}
using UnityEngine;

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
    public int enemyCount = 2;
    
    [Header("스폰 위치")]
    public Transform[] playerSpawnPoints;
    public Transform[] enemySpawnPoints;
    
    [Header("자동 생성 설정")]
    public bool createDummyDataIfEmpty = true;
    public bool createGridSystem = true;
    public Vector3[] defaultPlayerSpawns = {
        new Vector3(-3.5f, 0.5f, 0),  // 렉스 (격자 안쪽)
        new Vector3(-2.5f, 0.5f, 0),  // 루나
        new Vector3(-1.5f, 0.5f, 0),  // 제로
        new Vector3(0.5f, 1.5f, 0)    // 노바
    };
    public Vector3[] defaultEnemySpawns = {
        new Vector3(3.5f, 0.5f, 0),   // 적1 (격자 안쪽)
        new Vector3(4.5f, 1.5f, 0)    // 적2
    };
    
    private void Start()
    {
        if (autoStartBattle)
        {
            Invoke("InitializeAndStartBattle", battleStartDelay);
        }
    }
    
    public void InitializeAndStartBattle()
    {
        // 격자 시스템 생성
        if (createGridSystem)
        {
            CreateGridSystem();
        }
        
        // 더미 데이터 자동 생성
        if (createDummyDataIfEmpty)
        {
            CreateDummyDataIfNeeded();
        }
        
        // 플레이어 기계들 생성
        CreatePlayerMechs();
        
        // 적들 생성
        CreateEnemies();
        
        // 전투 시작
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
        
        // 더미 데이터를 사용할 때는 항상 기본 스폰 포인트 생성
        CreateDefaultSpawnPoints();
    }
    
    private GameObject CreateDummyMech(string name, MechType mechType)
    {
        GameObject dummyMech = new GameObject(name);
        dummyMech.transform.position = Vector3.zero;
        
        // 2D용 시각적 표현을 위한 쿼드 생성
        GameObject visualQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        visualQuad.name = "VisualQuad";
        visualQuad.transform.parent = dummyMech.transform;
        visualQuad.transform.localPosition = Vector3.zero;
        visualQuad.transform.localScale = new Vector3(1f, 1f, 1); // 격자 한 칸과 똑같은 크기
        
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
        dummyEnemy.transform.position = Vector3.zero;
        
        // 2D용 시각적 표현을 위한 쿼드 생성
        GameObject visualQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        visualQuad.name = "VisualQuad";
        visualQuad.transform.parent = dummyEnemy.transform;
        visualQuad.transform.localPosition = Vector3.zero;
        visualQuad.transform.localScale = new Vector3(1f, 1f, 1); // 격자 한 칸과 똑같은 크기
        
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
        playerSpawnPoints = new Transform[defaultPlayerSpawns.Length];
        
        for (int i = 0; i < defaultPlayerSpawns.Length; i++)
        {
            GameObject spawnPoint = new GameObject($"PlayerSpawn_{i}");
            spawnPoint.transform.parent = playerSpawnParent.transform;
            spawnPoint.transform.position = defaultPlayerSpawns[i];
            playerSpawnPoints[i] = spawnPoint.transform;
        }
        
        // 적 스폰 포인트 생성
        GameObject enemySpawnParent = new GameObject("EnemySpawnPoints");
        enemySpawnPoints = new Transform[defaultEnemySpawns.Length];
        
        for (int i = 0; i < defaultEnemySpawns.Length; i++)
        {
            GameObject spawnPoint = new GameObject($"EnemySpawn_{i}");
            spawnPoint.transform.parent = enemySpawnParent.transform;
            spawnPoint.transform.position = defaultEnemySpawns[i];
            enemySpawnPoints[i] = spawnPoint.transform;
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
        for (int i = 0; i < enemyCount; i++)
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
}

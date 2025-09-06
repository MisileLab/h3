using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class BattleSystem : MonoBehaviour
{
    [Header("전투 설정")]
    public bool isBattleActive = false;
    public BattlePhase currentPhase = BattlePhase.Preparation;
    public int currentTurn = 0;
    public int currentActorIndex = 0;
    public float turnTimeLimit = 30f;           // 턴 제한 시간 (초)
    public float currentTurnTimer = 0f;         // 현재 턴 타이머
    
    [Header("참가자")]
    public List<MechCharacter> playerMechs = new List<MechCharacter>();
    public List<EnemyAI> enemies = new List<EnemyAI>();
    public List<BattleActor> turnOrder = new List<BattleActor>();
    
    [Header("전투 옵션")]
    public bool allowRetreat = true;
    public bool allowNegotiation = true;
    public float retreatChance = 0.8f;
    
    [Header("협력 시스템")]
    public CooperativeActionManager cooperativeManager;
    
    [Header("전투 통계")]
    public int totalTurns = 0;
    public int playerCasualties = 0;
    public int enemyCasualties = 0;
    public bool perfectProtection = true;       // 아무도 다치지 않았는지
    
    // 이벤트
    public static event System.Action<BattleSystem> OnBattleStart;
    public static event System.Action<BattleSystem> OnBattleEnd;
    public static event System.Action<BattleSystem, BattleActor> OnTurnStart;
    public static event System.Action<BattleSystem, BattleActor> OnTurnEnd;
    
    private void Start()
    {
        InitializeBattle();
    }
    
    private void Update()
    {
        if (isBattleActive)
        {
            UpdateBattle();
            UpdateTurnTimer();
        }
    }
    
    private void InitializeBattle()
    {
        // 협력 시스템 초기화
        if (cooperativeManager == null)
        {
            cooperativeManager = FindObjectOfType<CooperativeActionManager>();
            if (cooperativeManager == null)
            {
                GameObject coopManager = new GameObject("CooperativeActionManager");
                cooperativeManager = coopManager.AddComponent<CooperativeActionManager>();
            }
        }
        
        // 플레이어 기계들 찾기
        playerMechs.Clear();
        MechCharacter[] foundMechs = FindObjectsOfType<MechCharacter>();
        foreach (MechCharacter mech in foundMechs)
        {
            if (mech.isAlive)
            {
                playerMechs.Add(mech);
                
                // 이벤트 구독
                MechCharacter.OnMechIncapacitated += OnPlayerIncapacitated;
            }
        }
        
        // 적들 찾기
        enemies.Clear();
        EnemyAI[] foundEnemies = FindObjectsOfType<EnemyAI>();
        foreach (EnemyAI enemy in foundEnemies)
        {
            if (enemy.isAlive)
            {
                enemies.Add(enemy);
            }
        }
        
        // 전투 통계 초기화
        totalTurns = 0;
        playerCasualties = 0;
        enemyCasualties = 0;
        perfectProtection = true;
        
        Debug.Log($"전투 초기화: 플레이어 {playerMechs.Count}명, 적 {enemies.Count}명");
        Debug.Log("협력형 전투 시스템 로드 완료!");
    }
    
    public void StartBattle()
    {
        if (isBattleActive) return;
        
        isBattleActive = true;
        currentPhase = BattlePhase.Active;
        currentTurn = 1;
        
        // 턴 순서 결정
        DetermineTurnOrder();
        
        // 첫 번째 행동자로 시작
        currentActorIndex = 0;
        
        OnBattleStart?.Invoke(this);
        Debug.Log("전투가 시작되었습니다!");
        
        StartNextTurn();
    }
    
    private void DetermineTurnOrder()
    {
        turnOrder.Clear();
        
        // 플레이어 기계들 추가
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive)
            {
                turnOrder.Add(new BattleActor(mech, mech.stats.speed + Random.Range(0, 10)));
            }
        }
        
        // 적들 추가
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive)
            {
                turnOrder.Add(new BattleActor(enemy, enemy.speed + Random.Range(0, 10)));
            }
        }
        
        // 속도 순으로 정렬
        turnOrder = turnOrder.OrderByDescending(actor => actor.initiative).ToList();
        
        Debug.Log("턴 순서 결정 완료:");
        for (int i = 0; i < turnOrder.Count; i++)
        {
            Debug.Log($"{i + 1}. {turnOrder[i].GetName()} (속도: {turnOrder[i].initiative})");
        }
    }
    
    private void UpdateBattle()
    {
        // 전투 종료 조건 확인
        if (CheckBattleEndConditions())
        {
            EndBattle();
            return;
        }
        
        // 현재 행동자 업데이트
        if (currentActorIndex < turnOrder.Count)
        {
            BattleActor currentActor = turnOrder[currentActorIndex];
            
            if (currentActor.IsAlive())
            {
                // AI 행동 처리 (플레이어는 직접 조작)
                if (currentActor.isEnemy)
                {
                    // 적 AI의 턴이면 자동으로 행동 수행
                    if (currentActor.enemy.PerformAction())
                    {
                        // 행동 완료 후 턴 종료
                        Invoke("EndCurrentTurn", 1f); // 1초 후 턴 종료
                    }
                }
            }
            else
            {
                // 죽은 행동자는 턴 스킵
                Debug.Log($"{currentActor.GetName()}이 전투 불능 상태입니다. 턴을 스킵합니다.");
                EndCurrentTurn();
            }
        }
    }
    
    /// <summary>
    /// 턴 타이머 업데이트
    /// </summary>
    private void UpdateTurnTimer()
    {
        if (currentTurnTimer > 0)
        {
            currentTurnTimer -= Time.deltaTime;
            
            if (currentTurnTimer <= 0)
            {
                // 시간 초과 시 자동으로 턴 종료
                Debug.Log("턴 시간 초과! 자동으로 턴을 종료합니다.");
                EndCurrentTurn();
            }
        }
    }
    
    public void EndCurrentTurn()
    {
        if (currentActorIndex < turnOrder.Count)
        {
            BattleActor currentActor = turnOrder[currentActorIndex];
            OnTurnEnd?.Invoke(this, currentActor);
            
            if (currentActor.isEnemy)
            {
                currentActor.enemy.EndTurn();
            }
            else
            {
                currentActor.mech.EndTurn();
            }
        }
        
        // 다음 행동자로 이동
        currentActorIndex++;
        
        // 모든 행동자의 턴이 끝났으면 새 턴 시작
        if (currentActorIndex >= turnOrder.Count)
        {
            currentTurn++;
            currentActorIndex = 0;
            
            // 턴 종료 시 처리
            ProcessEndOfTurn();
        }
        
        StartNextTurn();
    }
    
    private void StartNextTurn()
    {
        if (currentActorIndex < turnOrder.Count)
        {
            BattleActor currentActor = turnOrder[currentActorIndex];
            
            if (currentActor.IsAlive())
            {
                OnTurnStart?.Invoke(this, currentActor);
                
                // 턴 타이머 설정
                currentTurnTimer = turnTimeLimit;
                
                if (currentActor.isEnemy)
                {
                    currentActor.enemy.StartTurn();
                    Debug.Log($"<color=red>턴 {currentTurn}: {currentActor.GetName()}의 차례</color>");
                }
                else
                {
                    currentActor.mech.StartTurn();
                    Debug.Log($"<color=blue>턴 {currentTurn}: {currentActor.GetName()}의 차례 (AP: {currentActor.mech.actionPoints.currentAP})</color>");
                    
                    // 플레이어 턴 시작 시 사용 가능한 협력 행동 체크
                    CheckAvailableCooperativeActions(currentActor.mech);
                }
            }
            else
            {
                // 죽은 행동자는 턴 스킵하고 다음으로
                EndCurrentTurn();
            }
        }
    }
    
    private void ProcessEndOfTurn()
    {
        totalTurns++;
        
        // 모든 기계의 쿨다운 업데이트
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive)
            {
                mech.UpdateCooldowns(1f);
            }
        }
        
        // 협력 행동 쿨다운 업데이트
        if (cooperativeManager != null)
        {
            cooperativeManager.UpdateAllCooldowns();
        }
        
        // 상태 효과 업데이트
        UpdateStatusEffects();
        
        // 턴 종료 시 특별한 상호작용 체크
        CheckEndOfTurnInteractions();
        
        Debug.Log($"<color=gray>===== 턴 {currentTurn} 종료 =====</color>");
        LogBattleStatus();
    }
    
    private bool CheckBattleEndConditions()
    {
        // 플레이어 전멸 확인
        bool allPlayersDead = true;
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive)
            {
                allPlayersDead = false;
                break;
            }
        }
        
        if (allPlayersDead)
        {
            Debug.Log("플레이어 팀이 전멸했습니다. 전투 패배!");
            return true;
        }
        
        // 적 전멸 확인
        bool allEnemiesDead = true;
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive)
            {
                allEnemiesDead = false;
                break;
            }
        }
        
        if (allEnemiesDead)
        {
            Debug.Log("모든 적이 파괴되었습니다. 전투 승리!");
            return true;
        }
        
        return false;
    }
    
    private void EndBattle()
    {
        isBattleActive = false;
        currentPhase = BattlePhase.End;
        
        OnBattleEnd?.Invoke(this);
        Debug.Log("전투가 종료되었습니다.");
        
        // 전투 결과 처리
        ProcessBattleResult();
    }
    
    private void ProcessBattleResult()
    {
        // 승리/패배 판정
        bool victory = enemies.All(enemy => !enemy.isAlive);
        
        // 적 손실 카운트
        enemyCasualties = enemies.Count(enemy => !enemy.isAlive);
        
        // 보상 및 결과 처리
        CalculateBattleRewards(victory);
        
        if (victory)
        {
            Debug.Log("<color=green>\"진정한 승리는 모두 함께 집에 돌아가는 것이다.\"</color>");
            
            // 생존자들의 최종 대화
            StartCoroutine(PlayVictoryDialogue());
        }
        else
        {
            Debug.Log("<color=yellow>패배했지만 포기하지 않는다. 이는 새로운 시작이다.</color>");
            
            // 후퇴 및 구출 시퀀스
            StartCoroutine(PlayDefeatDialogue());
        }
    }
    
    /// <summary>
    /// 승리 시 대화 시퀀스
    /// </summary>
    private IEnumerator PlayVictoryDialogue()
    {
        yield return new WaitForSeconds(1f);
        
        var aliveMechs = GetAlivePlayerMechs();
        
        // 각 기계가 순서대로 승리 소감 발표
        foreach (var mech in aliveMechs)
        {
            mech.SayDialogue(mech.GetRandomDialogue(DialogueType.Victory), DialogueType.Victory);
            yield return new WaitForSeconds(2f);
        }
        
        // 팀 전체가 함께하는 마지막 대사
        if (aliveMechs.Count > 1)
        {
            var leader = aliveMechs[0]; // 첫 번째 기계가 대표 발언
            leader.SayDialogue("모두 함께 집에 돌아가자!", DialogueType.Victory);
        }
    }
    
    /// <summary>
    /// 패배 시 대화 시퀀스
    /// </summary>
    private IEnumerator PlayDefeatDialogue()
    {
        yield return new WaitForSeconds(1f);
        
        var aliveMechs = GetAlivePlayerMechs();
        
        if (aliveMechs.Count > 0)
        {
            // 생존자가 있는 경우 - 구출 계획
            var survivor = aliveMechs[0];
            survivor.SayDialogue("동료들을 구출하러 돌아올 거야!", DialogueType.Determined);
            
            yield return new WaitForSeconds(2f);
            
            // 전투 불능 상태인 기계들을 위한 메시지
            var incapacitatedMechs = playerMechs.Where(m => !m.isAlive).ToList();
            if (incapacitatedMechs.Count > 0)
            {
                Debug.Log($"<color=yellow>AI 드라이브 회수 대상: {string.Join(", ", incapacitatedMechs.Select(m => m.mechName))}</color>");
                Debug.Log("그들의 기억과 경험은 다음 미션에서 살아날 것이다.");
            }
        }
        else
        {
            // 전멸한 경우 - 하지만 희망적인 메시지
            Debug.Log("모든 기계가 전투 불능이 되었지만, 이들의 용기는 영원히 기억될 것이다.");
            Debug.Log("백업 시스템이 활성화되어 모든 AI가 안전하게 보존되었다.");
        }
    }
    
    public void AttemptRetreat()
    {
        if (!allowRetreat || !isBattleActive) return;
        
        float retreatRoll = Random.Range(0f, 1f);
        if (retreatRoll < retreatChance)
        {
            Debug.Log("후퇴에 성공했습니다!");
            EndBattle();
        }
        else
        {
            Debug.Log("후퇴에 실패했습니다. 전투를 계속합니다.");
        }
    }
    
    public void AttemptNegotiation(EnemyAI target)
    {
        if (!allowNegotiation || !isBattleActive) return;
        
        if (target.CanBeNegotiated())
        {
            // 협상 로직 (루나의 협상 스킬과 연동)
            Debug.Log($"{target.enemyName}과 협상을 시도합니다.");
        }
        else
        {
            Debug.Log($"{target.enemyName}은 협상할 수 없습니다.");
        }
    }
    
    public BattleActor GetCurrentActor()
    {
        if (currentActorIndex < turnOrder.Count)
        {
            return turnOrder[currentActorIndex];
        }
        return null;
    }
    
    public List<MechCharacter> GetAlivePlayerMechs()
    {
        return playerMechs.Where(mech => mech.isAlive).ToList();
    }
    
    public List<EnemyAI> GetAliveEnemies()
    {
        return enemies.Where(enemy => enemy.isAlive).ToList();
    }
    
    /// <summary>
    /// 플레이어가 전투 불능 상태가 되었을 때 호출되는 이벤트 핸들러
    /// </summary>
    /// <param name="mech">전투 불능이 된 기계</param>
    private void OnPlayerIncapacitated(MechCharacter mech)
    {
        playerCasualties++;
        perfectProtection = false;
        
        // 동료들의 반응 대사 (50% 확률)
        if (UnityEngine.Random.Range(0f, 1f) < 0.5f)
        {
            var aliveMechs = GetAlivePlayerMechs();
            if (aliveMechs.Count > 0)
            {
                var worriedMech = aliveMechs[UnityEngine.Random.Range(0, aliveMechs.Count)];
                worriedMech.SayDialogue($"{mech.mechName}! 버텨!", DialogueType.Worried);
            }
        }
        
        Debug.Log($"<color=red>{mech.mechName}이 전투 불능이 되었습니다. 총 손실: {playerCasualties}명</color>");
    }
    
    /// <summary>
    /// 사용 가능한 협력 행동들을 체크하고 알림
    /// </summary>
    /// <param name="currentMech">현재 턴인 기계</param>
    private void CheckAvailableCooperativeActions(MechCharacter currentMech)
    {
        if (cooperativeManager == null) return;
        
        var allies = GetAlivePlayerMechs();
        var possiblePartners = allies.Where(m => m != currentMech).ToList();
        
        foreach (var partner in possiblePartners)
        {
            var actorPair = new List<MechCharacter> { currentMech, partner };
            var availableActions = cooperativeManager.GetAvailableActions(actorPair);
            
            if (availableActions.Count > 0)
            {
                Debug.Log($"<color=green>{currentMech.mechName}과 {partner.mechName}이 사용할 수 있는 협력 행동: {availableActions.Count}개</color>");
            }
        }
    }
    
    /// <summary>
    /// 상태 효과들을 업데이트합니다
    /// </summary>
    private void UpdateStatusEffects()
    {
        // 플레이어 기계들의 상태 효과 업데이트
        foreach (var mech in playerMechs)
        {
            if (mech.isAlive)
            {
                mech.UpdateStatuses();
            }
        }
        
        // 적들의 상태 효과 업데이트
        foreach (var enemy in enemies)
        {
            if (enemy.isAlive)
            {
                enemy.UpdateStatusEffects();
            }
        }
    }
    
    /// <summary>
    /// 턴 종료 시 특별한 상호작용을 체크합니다
    /// </summary>
    private void CheckEndOfTurnInteractions()
    {
        // 위험한 상황에 있는 기계 체크
        var criticalMechs = playerMechs.Where(m => m.isAlive && (float)m.stats.currentHP / m.stats.maxHP < 0.3f).ToList();
        
        foreach (var criticalMech in criticalMechs)
        {
            // 30% 확률로 동료가 걱정하는 대사
            if (UnityEngine.Random.Range(0f, 1f) < 0.3f)
            {
                var allies = playerMechs.Where(m => m != criticalMech && m.isAlive).ToList();
                if (allies.Count > 0)
                {
                    var worriedAlly = allies[UnityEngine.Random.Range(0, allies.Count)];
                    worriedAlly.SayDialogue($"{criticalMech.mechName}, 상태가 안 좋아 보여!", DialogueType.Worried);
                    
                    // 치료 가능한 상황이면 제안
                    if (worriedAlly is LunaMech luna && luna.actionPoints.CanUseAP(2))
                    {
                        luna.SayDialogue("다음 턴에 치료해줄게!", DialogueType.Cooperative);
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// 현재 전투 상태를 로그로 출력합니다
    /// </summary>
    private void LogBattleStatus()
    {
        int alivePlayerCount = GetAlivePlayerMechs().Count;
        int aliveEnemyCount = GetAliveEnemies().Count;
        
        Debug.Log($"전투 상황: 아군 {alivePlayerCount}/{playerMechs.Count}, 적군 {aliveEnemyCount}/{enemies.Count}");
        
        // 각 기계의 HP 상태
        foreach (var mech in playerMechs)
        {
            if (mech.isAlive)
            {
                float hpPercent = (float)mech.stats.currentHP / mech.stats.maxHP * 100;
                string statusColor = hpPercent > 70 ? "green" : hpPercent > 30 ? "yellow" : "red";
                Debug.Log($"<color={statusColor}>{mech.mechName}: HP {mech.stats.currentHP}/{mech.stats.maxHP} ({hpPercent:F1}%)</color>");
            }
        }
    }
    
    /// <summary>
    /// 플레이어 기계가 협력 행동을 시도합니다
    /// </summary>
    /// <param name="action">협력 행동</param>
    /// <param name="actors">행동자들</param>
    /// <param name="target">대상 (옵션)</param>
    /// <returns>성공하면 true</returns>
    public bool TryCooperativeAction(CooperativeAction action, List<MechCharacter> actors, object target = null)
    {
        if (cooperativeManager == null) return false;
        
        return action.Perform(actors, target);
    }
    
    /// <summary>
    /// 플레이어가 수동으로 턴을 종료합니다
    /// </summary>
    public void PlayerEndTurn()
    {
        if (!isBattleActive) return;
        
        var currentActor = GetCurrentActor();
        if (currentActor != null && !currentActor.isEnemy)
        {
            Debug.Log($"{currentActor.GetName()}이 수동으로 턴을 종료합니다.");
            EndCurrentTurn();
        }
    }
    
    /// <summary>
    /// 전투 결과에 따른 보상 계산
    /// </summary>
    /// <param name="victory">승리 여부</param>
    private void CalculateBattleRewards(bool victory)
    {
        if (victory)
        {
            Debug.Log("<color=green>=== 전투 승리! ===</color>");
            
            // 보상 계산
            int baseReward = 100;
            int bonusReward = 0;
            
            // 완벽한 보호 보너스 (아무도 다치지 않음)
            if (perfectProtection)
            {
                bonusReward += 50;
                Debug.Log("완벽한 보호 달성! 보너스 +50");
            }
            
            // 빠른 승리 보너스 (5턴 이내)
            if (totalTurns <= 5)
            {
                bonusReward += 30;
                Debug.Log("신속한 승리! 보너스 +30");
            }
            
            // 협력 행동 보너스
            bonusReward += cooperativeManager.availableActions.Count * 5;
            
            int totalReward = baseReward + bonusReward;
            Debug.Log($"총 보상: {totalReward}점 (기본 {baseReward} + 보너스 {bonusReward})");
            
            // 승리 대사
            var aliveMechs = GetAlivePlayerMechs();
            if (aliveMechs.Count > 0)
            {
                foreach (var mech in aliveMechs)
                {
                    if (UnityEngine.Random.Range(0f, 1f) < 0.5f) // 50% 확률
                    {
                        mech.SayDialogue(mech.GetRandomDialogue(DialogueType.Victory), DialogueType.Victory);
                    }
                }
            }
        }
        else
        {
            Debug.Log("<color=red>=== 전투 패배... ===</color>");
            Debug.Log("하지만 이는 끝이 아니다. 모든 기계의 AI 드라이브는 안전하게 회수되었다.");
            
            // 패배 대사
            var aliveMechs = GetAlivePlayerMechs();
            if (aliveMechs.Count > 0)
            {
                var lastMech = aliveMechs[UnityEngine.Random.Range(0, aliveMechs.Count)];
                lastMech.SayDialogue("후퇴! 모두 무사히 돌아가자!", DialogueType.Defeat);
            }
        }
        
        Debug.Log($"전투 통계: 총 {totalTurns}턴, 아군 손실 {playerCasualties}명, 적군 손실 {enemyCasualties}명");
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        MechCharacter.OnMechIncapacitated -= OnPlayerIncapacitated;
    }
}

[System.Serializable]
public class BattleActor
{
    public MechCharacter mech;
    public EnemyAI enemy;
    public bool isEnemy;
    public int initiative;
    
    public BattleActor(MechCharacter mechCharacter, int init)
    {
        mech = mechCharacter;
        enemy = null;
        isEnemy = false;
        initiative = init;
    }
    
    public BattleActor(EnemyAI enemyAI, int init)
    {
        mech = null;
        enemy = enemyAI;
        isEnemy = true;
        initiative = init;
    }
    
    public string GetName()
    {
        if (isEnemy)
        {
            return enemy.enemyName;
        }
        else
        {
            return mech.mechName;
        }
    }
    
    public bool IsAlive()
    {
        if (isEnemy)
        {
            return enemy.isAlive;
        }
        else
        {
            return mech.isAlive;
        }
    }
    
    public void StartTurn()
    {
        if (isEnemy)
        {
            enemy.StartTurn();
        }
        else
        {
            mech.StartTurn();
        }
    }
    
    public void EndTurn()
    {
        if (isEnemy)
        {
            enemy.EndTurn();
        }
        else
        {
            mech.EndTurn();
        }
    }
}

public enum BattlePhase
{
    Preparation,    // 전투 준비
    Active,         // 전투 진행 중
    End             // 전투 종료
}

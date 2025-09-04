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
    
    [Header("참가자")]
    public List<MechCharacter> playerMechs = new List<MechCharacter>();
    public List<EnemyAI> enemies = new List<EnemyAI>();
    public List<BattleActor> turnOrder = new List<BattleActor>();
    
    [Header("전투 옵션")]
    public bool allowRetreat = true;
    public bool allowNegotiation = true;
    public float retreatChance = 0.8f;
    
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
        }
    }
    
    private void InitializeBattle()
    {
        // 플레이어 기계들 찾기
        playerMechs.Clear();
        MechCharacter[] foundMechs = FindObjectsOfType<MechCharacter>();
        foreach (MechCharacter mech in foundMechs)
        {
            if (mech.isAlive)
            {
                playerMechs.Add(mech);
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
        
        Debug.Log($"전투 초기화: 플레이어 {playerMechs.Count}명, 적 {enemies.Count}명");
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
                // AI 행동 처리
                if (currentActor.isEnemy)
                {
                    currentActor.enemy.PerformAction();
                }
            }
            else
            {
                // 죽은 행동자는 턴 스킵
                Debug.Log($"{currentActor.GetName()}이 전투 불능 상태입니다. 턴을 스킵합니다.");
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
                
                if (currentActor.isEnemy)
                {
                    currentActor.enemy.StartTurn();
                }
                else
                {
                    currentActor.mech.StartTurn();
                }
                
                Debug.Log($"턴 {currentTurn}: {currentActor.GetName()}의 차례");
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
        // 모든 기계의 쿨다운 업데이트
        foreach (MechCharacter mech in playerMechs)
        {
            if (mech.isAlive)
            {
                mech.UpdateCooldowns(1f);
            }
        }
        
        // 상태 효과 업데이트 등
        Debug.Log($"턴 {currentTurn} 종료");
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
        // 승리/패배에 따른 보상 및 페널티 처리
        bool victory = enemies.All(enemy => !enemy.isAlive);
        
        if (victory)
        {
            Debug.Log("승리! 보상을 획득했습니다.");
            // 보상 시스템 구현
        }
        else
        {
            Debug.Log("패배! 후퇴합니다.");
            // 후퇴 처리
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

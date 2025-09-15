using UnityEngine;
using System;

public class ActionModeManager : MonoBehaviour
{
    [Header("Action Settings")]
    public int defaultAttackRange = 2;
    public int defaultMoveRange = 3;
    public int defaultSkillRange = 2;
    
    [Header("Debug")]
    public bool debugMode = false;
    
    // 현재 행동 모드 상태
    public enum ActionMode
    {
        None,       // 행동 모드 없음
        Attack,     // 공격 모드
        Move,       // 이동 모드
        Skill,      // 스킬 모드
        Defend      // 방어 모드 (즉시 실행)
    }
    
    public ActionMode currentMode = ActionMode.None;
    private MechCharacter currentActor;
    
    // 시스템 참조
    private RangeDisplay rangeDisplay;
    private TargetSelector targetSelector;
    private BattleGridManager gridManager;
    private BattleUI battleUI;
    
    // 이벤트
    public static event Action<ActionMode> OnActionModeChanged;
    public static event Action<MechCharacter, int, int, ActionMode> OnActionExecuted;
    
    private void Start()
    {
        // 시스템 참조 가져오기
        rangeDisplay = FindObjectOfType<RangeDisplay>();
        targetSelector = FindObjectOfType<TargetSelector>();
        gridManager = FindObjectOfType<BattleGridManager>();
        battleUI = FindObjectOfType<BattleUI>();
        
        if (rangeDisplay == null)
        {
            Debug.LogError("RangeDisplay를 찾을 수 없습니다!");
        }
        
        if (targetSelector == null)
        {
            Debug.LogError("TargetSelector를 찾을 수 없습니다!");
        }
        
        // 타겟 선택 이벤트 구독
        TargetSelector.OnTargetSelected += OnTargetSelected;
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        TargetSelector.OnTargetSelected -= OnTargetSelected;
    }
    
    /// <summary>
    /// 행동 모드를 설정합니다
    /// </summary>
    public void SetActionMode(ActionMode mode, MechCharacter actor)
    {
        if (actor == null)
        {
            Debug.LogWarning("SetActionMode: actor가 null입니다!");
            return;
        }
        
        currentMode = mode;
        currentActor = actor;
        
        if (debugMode)
        {
            Debug.Log($"행동 모드 변경: {mode} (행동자: {actor.mechName})");
        }
        
        // 이전 모드 정리
        ClearCurrentMode();
        
        // 새로운 모드 시작
        switch (mode)
        {
            case ActionMode.Attack:
                StartAttackMode();
                break;
            case ActionMode.Move:
                StartMoveMode();
                break;
            case ActionMode.Skill:
                StartSkillMode();
                break;
            case ActionMode.Defend:
                ExecuteDefend();
                break;
            case ActionMode.None:
                // 아무것도 하지 않음
                break;
        }
        
        // 모드 변경 이벤트 발생
        OnActionModeChanged?.Invoke(mode);
    }
    
    private void StartAttackMode()
    {
        if (currentActor == null) return;
        
        // AP 체크
        if (currentActor.actionPoints.currentAP < 1)
        {
            battleUI?.ShowDialogue(currentActor.mechName, "AP가 부족해서 공격할 수 없다!");
            SetActionMode(ActionMode.None, currentActor);
            return;
        }
        
        // 공격 사정거리 표시
        int attackRange = GetActorAttackRange(currentActor);
        rangeDisplay?.ShowAttackRange(currentActor.transform.position, attackRange);
        
        // 타겟 선택 모드 시작
        targetSelector?.StartTargetSelection(RangeDisplay.RangeType.Attack);
        
        if (debugMode)
        {
            Debug.Log($"공격 모드 시작 - 사정거리: {attackRange}");
        }
    }
    
    private void StartMoveMode()
    {
        if (currentActor == null) return;
        
        // AP 체크
        if (currentActor.actionPoints.currentAP < 1)
        {
            battleUI?.ShowDialogue(currentActor.mechName, "AP가 부족해서 이동할 수 없다!");
            SetActionMode(ActionMode.None, currentActor);
            return;
        }
        
        // 이동 범위 표시
        int moveRange = GetActorMoveRange(currentActor);
        Vector3 actorPosition = currentActor.transform.position;
        
        Debug.Log($"이동 모드 시작 - 기계 위치: {actorPosition}, 이동 범위: {moveRange}");
        rangeDisplay?.ShowMoveRange(actorPosition, moveRange);
        
        // 타겟 선택 모드 시작
        targetSelector?.StartTargetSelection(RangeDisplay.RangeType.Move);
        
        if (debugMode)
        {
            Debug.Log($"이동 모드 시작 - 이동 범위: {moveRange}");
        }
    }
    
    private void StartSkillMode()
    {
        if (currentActor == null) return;
        
        // 스킬 사용 가능한지 체크
        if (!CanUseSkill(currentActor))
        {
            battleUI?.ShowDialogue(currentActor.mechName, "스킬을 사용할 수 없다!");
            SetActionMode(ActionMode.None, currentActor);
            return;
        }
        
        // 스킬 범위 표시
        int skillRange = GetActorSkillRange(currentActor);
        rangeDisplay?.ShowSkillRange(currentActor.transform.position, skillRange);
        
        // 타겟 선택 모드 시작
        targetSelector?.StartTargetSelection(RangeDisplay.RangeType.Skill);
        
        if (debugMode)
        {
            Debug.Log($"스킬 모드 시작 - 스킬 범위: {skillRange}");
        }
    }
    
    private void ExecuteDefend()
    {
        if (currentActor == null) return;
        
        // AP 체크
        if (currentActor.actionPoints.currentAP < 1)
        {
            battleUI?.ShowDialogue(currentActor.mechName, "AP가 부족해서 방어할 수 없다!");
            SetActionMode(ActionMode.None, currentActor);
            return;
        }
        
        // 방어 실행
        currentActor.isGuarding = true;
        currentActor.ConsumeAP(1);
        
        battleUI?.ShowDialogue(currentActor.mechName, "방어 태세를 취한다!");
        
        if (debugMode)
        {
            Debug.Log($"{currentActor.mechName}이(가) 방어를 시작했습니다.");
        }
        
        // 행동 완료
        OnActionExecuted?.Invoke(currentActor, 0, 0, ActionMode.Defend);
        SetActionMode(ActionMode.None, currentActor);
    }
    
    /// <summary>
    /// 타겟이 선택되었을 때 호출되는 콜백
    /// </summary>
    private void OnTargetSelected(int gridX, int gridY, RangeDisplay.RangeType rangeType)
    {
        if (currentActor == null) return;
        
        Vector3 targetWorldPos = targetSelector.GridToWorldPosition(gridX, gridY);
        
        if (debugMode)
        {
            Debug.Log($"타겟 선택됨: 격자({gridX}, {gridY}), 월드({targetWorldPos}), 타입: {rangeType}");
        }
        
        // 범위 타입에 따른 행동 실행
        switch (rangeType)
        {
            case RangeDisplay.RangeType.Attack:
                ExecuteAttack(gridX, gridY, targetWorldPos);
                break;
            case RangeDisplay.RangeType.Move:
                ExecuteMove(gridX, gridY, targetWorldPos);
                break;
            case RangeDisplay.RangeType.Skill:
                ExecuteSkill(gridX, gridY, targetWorldPos);
                break;
        }
    }
    
    private void ExecuteAttack(int gridX, int gridY, Vector3 targetWorldPos)
    {
        // 타겟 위치의 적 찾기
        EnemyAI targetEnemy = FindEnemyAtPosition(targetWorldPos);
        
        if (targetEnemy != null && targetEnemy.isAlive)
        {
            // 공격 실행
            float damage = currentActor.stats.attack;
            string attackMessage = $"{currentActor.mechName}이(가) {targetEnemy.enemyName}을(를) {damage} 데미지로 공격!";
            
            Debug.Log(attackMessage);
            battleUI?.ShowDialogue(currentActor.mechName, "공격한다!");
            
            targetEnemy.TakeDamage(damage);
            currentActor.ConsumeAP(1);
            
            // 행동 완료 이벤트
            OnActionExecuted?.Invoke(currentActor, gridX, gridY, ActionMode.Attack);
        }
        else
        {
            Debug.LogWarning("공격할 대상을 찾을 수 없습니다!");
            battleUI?.ShowDialogue("시스템", "공격할 대상이 없습니다!");
        }
        
        // 공격 모드 종료
        SetActionMode(ActionMode.None, currentActor);
    }
    
    private void ExecuteMove(int gridX, int gridY, Vector3 targetWorldPos)
    {
        // 이동 실행
        currentActor.transform.position = targetWorldPos;
        currentActor.ConsumeAP(1);
        
        battleUI?.ShowDialogue(currentActor.mechName, "이동한다!");
        
        if (debugMode)
        {
            Debug.Log($"{currentActor.mechName}이(가) {targetWorldPos}로 이동했습니다.");
        }
        
        // 행동 완료 이벤트
        OnActionExecuted?.Invoke(currentActor, gridX, gridY, ActionMode.Move);
        
        // 이동 모드 종료
        SetActionMode(ActionMode.None, currentActor);
    }
    
    private void ExecuteSkill(int gridX, int gridY, Vector3 targetWorldPos)
    {
        // 스킬 실행 (기계 타입별로 다르게 처리)
        bool skillExecuted = false;
        
        if (currentActor.mechType == MechType.Rex)
        {
            RexMech rex = currentActor as RexMech;
            if (rex != null)
            {
                rex.UseGuardianShield();
                skillExecuted = true;
            }
        }
        else if (currentActor.mechType == MechType.Luna)
        {
            LunaMech luna = currentActor as LunaMech;
            if (luna != null)
            {
                // 타겟 위치의 아군 찾기
                MechCharacter targetMech = FindMechAtPosition(targetWorldPos);
                if (targetMech != null)
                {
                    luna.UseNanoRepair(targetMech, BodyPartType.Torso);
                    skillExecuted = true;
                }
            }
        }
        
        if (skillExecuted)
        {
            battleUI?.ShowDialogue(currentActor.mechName, "스킬을 사용한다!");
            
            // 행동 완료 이벤트
            OnActionExecuted?.Invoke(currentActor, gridX, gridY, ActionMode.Skill);
        }
        else
        {
            battleUI?.ShowDialogue(currentActor.mechName, "스킬을 사용할 수 없다!");
        }
        
        // 스킬 모드 종료
        SetActionMode(ActionMode.None, currentActor);
    }
    
    /// <summary>
    /// 현재 모드를 정리합니다
    /// </summary>
    private void ClearCurrentMode()
    {
        rangeDisplay?.ClearAll();
        targetSelector?.EndTargetSelection();
    }
    
    /// <summary>
    /// 행동 모드를 취소합니다
    /// </summary>
    public void CancelCurrentMode()
    {
        if (debugMode)
        {
            Debug.Log("행동 모드 취소됨");
        }
        
        SetActionMode(ActionMode.None, currentActor);
    }
    
    // 헬퍼 메서드들
    private int GetActorAttackRange(MechCharacter actor)
    {
        // 기계별 고유 공격 사정거리가 있다면 여기서 처리
        return defaultAttackRange;
    }
    
    private int GetActorMoveRange(MechCharacter actor)
    {
        // 기계별 고유 이동 범위가 있다면 여기서 처리
        return defaultMoveRange;
    }
    
    private int GetActorSkillRange(MechCharacter actor)
    {
        // 기계별 고유 스킬 범위가 있다면 여기서 처리
        return defaultSkillRange;
    }
    
    private bool CanUseSkill(MechCharacter actor)
    {
        // 스킬 사용 조건 체크
        return actor.actionPoints.currentAP >= 1;
    }
    
    private EnemyAI FindEnemyAtPosition(Vector3 position)
    {
        EnemyAI[] enemies = FindObjectsOfType<EnemyAI>();
        foreach (EnemyAI enemy in enemies)
        {
            if (enemy.isAlive && Vector3.Distance(enemy.transform.position, position) < gridManager.gridSize * 0.6f)
            {
                return enemy;
            }
        }
        return null;
    }
    
    private MechCharacter FindMechAtPosition(Vector3 position)
    {
        MechCharacter[] mechs = FindObjectsOfType<MechCharacter>();
        foreach (MechCharacter mech in mechs)
        {
            if (mech.isAlive && Vector3.Distance(mech.transform.position, position) < gridManager.gridSize * 0.6f)
            {
                return mech;
            }
        }
        return null;
    }
    
    /// <summary>
    /// 현재 행동 중인 액터를 반환합니다
    /// </summary>
    public MechCharacter GetCurrentActor()
    {
        return currentActor;
    }
    
    /// <summary>
    /// 현재 행동 모드를 반환합니다
    /// </summary>
    public ActionMode GetCurrentMode()
    {
        return currentMode;
    }
}

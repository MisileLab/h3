using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class CooperationOptionUI : MonoBehaviour
{
    [Header("UI 요소")]
    public Text mechNameText;
    public Text skillNameText;
    public Text skillDescriptionText;
    public Text apCostText;
    public Button cooperationButton;
    public Image mechIcon;
    public Slider trustSlider;
    public Text trustText;
    
    private MechCharacter user;
    private MechCharacter target;
    private BattleUI battleUI;
    private CooperativeAction selectedAction;
    
    public void Initialize(MechCharacter userMech, MechCharacter targetMech, BattleUI ui)
    {
        user = userMech;
        target = targetMech;
        battleUI = ui;
        
        SetupUI();
        UpdateCooperationOptions();
    }
    
    private void SetupUI()
    {
        if (mechNameText != null)
        {
            mechNameText.text = target.mechName;
        }
        
        if (cooperationButton != null)
        {
            cooperationButton.onClick.AddListener(OnCooperationClicked);
        }
        
        UpdateTrustDisplay();
    }
    
    private void UpdateTrustDisplay()
    {
        // 신뢰도 표시
        if (trustSlider != null && trustText != null)
        {
            int trustLevel = GetTrustLevel();
            trustSlider.value = trustLevel / 100f;
            trustText.text = $"신뢰도: {trustLevel}";
        }
    }
    
    private void UpdateCooperationOptions()
    {
        if (user == null || target == null) return;
        
        // 협력 행동 매니저 찾기
        var cooperativeManager = FindObjectOfType<CooperativeActionManager>();
        if (cooperativeManager == null) 
        {
            DisplayNoActionsAvailable();
            return;
        }
        
        // 사용 가능한 협력 행동 찾기
        var actors = new List<MechCharacter> { user, target };
        var availableActions = cooperativeManager.GetAvailableActions(actors);
        
        if (availableActions.Count > 0)
        {
            selectedAction = availableActions[0]; // 첫 번째 행동을 기본 선택
            DisplayActionInfo();
        }
        else
        {
            DisplayNoActionsAvailable();
        }
    }
    
    private void DisplayActionInfo()
    {
        if (selectedAction == null) return;
        
        if (skillNameText != null)
        {
            skillNameText.text = selectedAction.actionName;
        }
        
        if (skillDescriptionText != null)
        {
            skillDescriptionText.text = selectedAction.description;
        }
        
        if (apCostText != null)
        {
            apCostText.text = $"AP: {selectedAction.apCost}";
        }
        
        // 사용 가능 여부에 따라 버튼 활성화
        if (cooperationButton != null)
        {
            bool canUse = user.actionPoints.CanUseAP(selectedAction.apCost) && 
                         target.actionPoints.CanUseAP(selectedAction.apCost);
            cooperationButton.interactable = canUse;
        }
    }
    
    private void DisplayNoActionsAvailable()
    {
        if (skillNameText != null)
        {
            skillNameText.text = "사용 불가";
        }
        
        if (skillDescriptionText != null)
        {
            skillDescriptionText.text = "현재 사용할 수 있는 협력 행동이 없습니다.";
        }
        
        if (apCostText != null)
        {
            apCostText.text = "";
        }
        
        if (cooperationButton != null)
        {
            cooperationButton.interactable = false;
        }
    }
    
    private void OnCooperationClicked()
    {
        if (user == null || target == null || selectedAction == null) return;
        
        // 협력 행동 실행
        var actors = new List<MechCharacter> { user, target };
        bool success = selectedAction.Perform(actors);
        
        if (success)
        {
            Debug.Log($"{user.mechName}과 {target.mechName}이 {selectedAction.actionName}을 사용했습니다!");
            
            // UI 업데이트
            if (battleUI != null)
            {
                // 선택된 협력 사용 후 패널을 닫고 버튼 상태 갱신
                battleUI.HideCooperationPanel();
            }
            
            // 전투 시스템에 턴 종료 신호 (AP가 부족한 경우)
            if (user.actionPoints.currentAP <= 0)
            {
                var battleSystem = FindObjectOfType<BattleSystem>();
                if (battleSystem != null)
                {
                    battleSystem.PlayerEndTurn();
                }
            }
        }
        else
        {
            Debug.Log($"협력 행동 사용에 실패했습니다.");
        }
    }
    
    public void OnMechClicked()
    {
        // 기계 클릭 시 상세 정보 표시
        if (target != null)
        {
            ShowMechDetails();
        }
    }
    
    private void ShowMechDetails()
    {
        string details = $"{target.mechName} 상세 정보\n";
        details += $"HP: {target.stats.currentHP}/{target.stats.maxHP}\n";
        details += $"AP: {target.actionPoints.currentAP}/{target.actionPoints.maxAP}\n";
        details += $"공격력: {target.stats.attack}\n";
        details += $"방어력: {target.stats.defense}\n";
        details += $"속도: {target.stats.speed}\n";
        
        // 부위 상태
        details += "\n부위 상태:\n";
        foreach (var part in target.bodyParts)
        {
            string status = part.isDestroyed ? "파괴" : $"{part.currentHP}/{part.maxHP}";
            details += $"- {part.partName}: {status}\n";
        }
        
        Debug.Log(details);
    }
    
    private int GetTrustLevel()
    {
        if (user == null || target == null) return 0;
        
        // 신뢰도 시스템과 연동
        if (user.trustLevels != null && user.trustLevels.ContainsKey(target))
        {
            return user.trustLevels[target];
        }
        
        return 0;
    }
}
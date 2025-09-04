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
    private CooperationSkill selectedSkill;
    
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
        
        // 신뢰도 표시
        if (trustSlider != null && trustText != null)
        {
            int trustLevel = user.trustLevels.ContainsKey(target.mechType) ? user.trustLevels[target.mechType] : 0;
            trustSlider.value = trustLevel / 100f;
            trustText.text = $"신뢰도: {trustLevel}";
        }
    }
    
    private void UpdateCooperationOptions()
    {
        if (user == null || target == null) return;
        
        // 사용 가능한 협력 스킬 찾기
        CooperationSystem coopSystem = CooperationSystem.Instance;
        if (coopSystem == null) return;
        
        List<CooperationSkill> availableSkills = coopSystem.GetAvailableSkillsForMech(user);
        
        // 첫 번째 사용 가능한 스킬을 기본으로 선택
        foreach (CooperationSkill skill in availableSkills)
        {
            if (coopSystem.CanUseCooperation(user, target, skill.skillName))
            {
                selectedSkill = skill;
                break;
            }
        }
        
        if (selectedSkill != null)
        {
            UpdateSkillInfo(selectedSkill);
        }
        else
        {
            // 사용 가능한 협력 스킬이 없음
            if (skillNameText != null)
            {
                skillNameText.text = "사용 불가";
            }
            
            if (skillDescriptionText != null)
            {
                skillDescriptionText.text = "협력할 수 없습니다.";
            }
            
            if (cooperationButton != null)
            {
                cooperationButton.interactable = false;
            }
        }
    }
    
    private void UpdateSkillInfo(CooperationSkill skill)
    {
        if (skillNameText != null)
        {
            skillNameText.text = skill.skillName;
        }
        
        if (skillDescriptionText != null)
        {
            skillDescriptionText.text = skill.description;
        }
        
        if (apCostText != null)
        {
            apCostText.text = $"AP: {skill.apCost}";
        }
        
        if (cooperationButton != null)
        {
            cooperationButton.interactable = user.stats.currentAP >= skill.apCost;
        }
    }
    
    private void OnCooperationClicked()
    {
        if (user == null || target == null || selectedSkill == null) return;
        
        // 협력 스킬 사용
        CooperationSystem coopSystem = CooperationSystem.Instance;
        if (coopSystem != null)
        {
            coopSystem.UseCooperation(user, target, selectedSkill.skillName);
        }
        
        // UI 업데이트
        if (battleUI != null)
        {
            battleUI.HideCooperationPanel();
        }
        
        // 전투 시스템에 턴 종료 신호 (필요한 경우)
        BattleSystem battleSystem = FindObjectOfType<BattleSystem>();
        if (battleSystem != null && user.stats.currentAP <= 0)
        {
            battleSystem.EndCurrentTurn();
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
        details += $"AP: {target.stats.currentAP}/{target.stats.maxAP}\n";
        details += $"공격력: {target.stats.attack}\n";
        details += $"방어력: {target.stats.defense}\n";
        details += $"속도: {target.stats.speed}\n";
        
        // 부위 상태
        details += "\n부위 상태:\n";
        foreach (MechBodyPart part in target.bodyParts)
        {
            string status = part.isDestroyed ? "파괴" : (part.isDamaged ? "손상" : "정상");
            details += $"- {part.partName}: {status}\n";
        }
        
        Debug.Log(details);
    }
}

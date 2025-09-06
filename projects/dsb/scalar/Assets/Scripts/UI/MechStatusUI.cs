using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class MechStatusUI : MonoBehaviour
{
    [Header("UI 요소")]
    public Text mechNameText;
    public Slider hpSlider;
    public Text hpText;
    public Slider apSlider;
    public Text apText;
    public Image mechIcon;
    public GameObject statusEffectsPanel;
    
    [Header("부위 상태")]
    public Transform bodyPartsContainer;
    public GameObject bodyPartStatusPrefab;
    
    [Header("상태 표시")]
    public Image guardIndicator;
    public Image stealthIndicator;
    public Image damageIndicator;
    
    private MechCharacter mech;
    private List<BodyPartStatusUI> bodyPartUIs = new List<BodyPartStatusUI>();
    
    public void Initialize(MechCharacter mechCharacter)
    {
        mech = mechCharacter;
        CreateBodyPartUIs();
        UpdateStatus();
    }
    
    private void CreateBodyPartUIs()
    {
        if (mech == null || bodyPartStatusPrefab == null || bodyPartsContainer == null) return;
        
        // 기존 UI 제거
        foreach (BodyPartStatusUI ui in bodyPartUIs)
        {
            if (ui != null) Destroy(ui.gameObject);
        }
        bodyPartUIs.Clear();
        
        // 부위별 UI 생성
        foreach (BodyPart part in mech.bodyParts)
        {
            GameObject partUI = Instantiate(bodyPartStatusPrefab, bodyPartsContainer);
            BodyPartStatusUI partStatusUI = partUI.GetComponent<BodyPartStatusUI>();
            if (partStatusUI != null)
            {
                partStatusUI.Initialize(part);
                bodyPartUIs.Add(partStatusUI);
            }
        }
    }
    
    public void UpdateStatus()
    {
        if (mech == null) return;
        
        // 기본 정보 업데이트
        if (mechNameText != null)
        {
            mechNameText.text = mech.mechName;
        }
        
        // HP 표시
        if (hpSlider != null)
        {
            hpSlider.value = (float)mech.stats.currentHP / mech.stats.maxHP;
        }
        
        if (hpText != null)
        {
            hpText.text = $"{mech.stats.currentHP}/{mech.stats.maxHP}";
        }
        
        // AP 표시
        if (apSlider != null)
        {
            apSlider.value = mech.actionPoints != null ? mech.actionPoints.GetAPRatio() : 0f;
        }
        
        if (apText != null)
        {
            apText.text = mech.actionPoints != null ? $"{mech.actionPoints.currentAP}/{mech.actionPoints.maxAP}" : "0/0";
        }
        
        // 상태 표시
        if (guardIndicator != null)
        {
            guardIndicator.gameObject.SetActive(mech.isGuarding);
        }
        
        if (stealthIndicator != null)
        {
            stealthIndicator.gameObject.SetActive(mech.isInStealth);
        }
        
        if (damageIndicator != null)
        {
            damageIndicator.gameObject.SetActive(mech.stats.currentHP < mech.stats.maxHP * 0.5f);
        }
        
        // 부위 상태 업데이트
        foreach (BodyPartStatusUI partUI in bodyPartUIs)
        {
            if (partUI != null)
            {
                partUI.UpdateStatus();
            }
        }
        
        // 기계 아이콘 색상 변경 (상태에 따라)
        if (mechIcon != null)
        {
            if (!mech.isAlive)
            {
                mechIcon.color = Color.gray;
            }
            else if (mech.stats.currentHP < mech.stats.maxHP * 0.25f)
            {
                mechIcon.color = Color.red;
            }
            else if (mech.stats.currentHP < mech.stats.maxHP * 0.5f)
            {
                mechIcon.color = Color.yellow;
            }
            else
            {
                mechIcon.color = Color.white;
            }
        }
    }
    
    private void Update()
    {
        // 실시간 업데이트 (필요한 경우)
        if (mech != null && mech.isInCombat)
        {
            UpdateStatus();
        }
    }
}

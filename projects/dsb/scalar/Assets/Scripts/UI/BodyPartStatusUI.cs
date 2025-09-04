using UnityEngine;
using UnityEngine.UI;

public class BodyPartStatusUI : MonoBehaviour
{
    [Header("UI 요소")]
    public Text partNameText;
    public Slider hpSlider;
    public Text hpText;
    public Image partIcon;
    public GameObject destroyedIndicator;
    public GameObject damagedIndicator;
    
    [Header("색상 설정")]
    public Color healthyColor = Color.green;
    public Color damagedColor = Color.yellow;
    public Color criticalColor = Color.red;
    public Color destroyedColor = Color.gray;
    
    private MechBodyPart bodyPart;
    
    public void Initialize(MechBodyPart part)
    {
        bodyPart = part;
        UpdateStatus();
    }
    
    public void UpdateStatus()
    {
        if (bodyPart == null) return;
        
        // 부위 이름 표시
        if (partNameText != null)
        {
            partNameText.text = bodyPart.partName;
        }
        
        // HP 표시
        if (hpSlider != null)
        {
            hpSlider.value = bodyPart.GetHPPercentage();
        }
        
        if (hpText != null)
        {
            hpText.text = $"{(int)bodyPart.currentHP}/{(int)bodyPart.maxHP}";
        }
        
        // 상태 표시
        if (destroyedIndicator != null)
        {
            destroyedIndicator.SetActive(bodyPart.isDestroyed);
        }
        
        if (damagedIndicator != null)
        {
            damagedIndicator.SetActive(bodyPart.isDamaged && !bodyPart.isDestroyed);
        }
        
        // 색상 변경
        UpdateColor();
    }
    
    private void UpdateColor()
    {
        if (bodyPart == null) return;
        
        Color targetColor = healthyColor;
        
        switch (bodyPart.GetDamageLevel())
        {
            case DamageLevel.Minor:
                targetColor = healthyColor;
                break;
            case DamageLevel.Moderate:
                targetColor = damagedColor;
                break;
            case DamageLevel.Severe:
                targetColor = criticalColor;
                break;
            case DamageLevel.Critical:
                targetColor = criticalColor;
                break;
            case DamageLevel.Destroyed:
                targetColor = destroyedColor;
                break;
        }
        
        // HP 슬라이더 색상 변경
        if (hpSlider != null)
        {
            Image sliderFill = hpSlider.fillRect.GetComponent<Image>();
            if (sliderFill != null)
            {
                sliderFill.color = targetColor;
            }
        }
        
        // 부위 아이콘 색상 변경
        if (partIcon != null)
        {
            partIcon.color = targetColor;
        }
    }
    
    public void OnPartClicked()
    {
        if (bodyPart == null) return;
        
        // 부위 클릭 시 상세 정보 표시
        string info = $"{bodyPart.partName}\n";
        info += $"HP: {bodyPart.currentHP:F0}/{bodyPart.maxHP:F0}\n";
        info += $"상태: {GetStatusText()}\n";
        
        if (bodyPart.isDestroyed)
        {
            info += $"효과: {bodyPart.destructionEffect}";
        }
        
        Debug.Log(info);
        
        // 상세 정보 UI 표시 (구현 필요)
        ShowPartDetails(info);
    }
    
    private string GetStatusText()
    {
        if (bodyPart.isDestroyed) return "파괴됨";
        if (bodyPart.isDamaged) return "손상됨";
        return "정상";
    }
    
    private void ShowPartDetails(string details)
    {
        // 부위 상세 정보를 표시하는 UI 구현
        // 임시로 Debug.Log 사용
        Debug.Log($"부위 상세 정보:\n{details}");
    }
}

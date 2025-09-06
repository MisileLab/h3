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
    
    private BodyPart bodyPart;
    
    public void Initialize(BodyPart part)
    {
        bodyPart = part;
        UpdateStatus();
        
        // 이벤트 구독
        BodyPart.OnDamageLevelChanged += OnDamageLevelChanged;
        BodyPart.OnPartDestroyed += OnPartDestroyed;
    }
    
    public void UpdateStatus()
    {
        if (bodyPart == null) return;
        
        // 부위 이름 표시
        if (partNameText != null)
        {
            partNameText.text = bodyPart.partName;
        }
        
        // HP 슬라이더 업데이트
        if (hpSlider != null)
        {
            float hpRatio = bodyPart.GetHPRatio();
            hpSlider.value = hpRatio;
            
            // HP 슬라이더 색상 변경
            Image fillImage = hpSlider.fillRect.GetComponent<Image>();
            if (fillImage != null)
            {
                fillImage.color = GetColorForDamageLevel(bodyPart.damageLevel);
            }
        }
        
        // HP 텍스트 업데이트
        if (hpText != null)
        {
            hpText.text = $"{bodyPart.currentHP}/{bodyPart.maxHP}";
        }
        
        // 상태 인디케이터 업데이트
        UpdateStatusIndicators();
    }
    
    private void UpdateStatusIndicators()
    {
        // 파괴 상태 표시
        if (destroyedIndicator != null)
        {
            destroyedIndicator.SetActive(bodyPart.isDestroyed);
        }
        
        // 손상 상태 표시
        if (damagedIndicator != null)
        {
            bool isDamaged = bodyPart.damageLevel != DamageLevel.None && !bodyPart.isDestroyed;
            damagedIndicator.SetActive(isDamaged);
        }
        
        // 부위 아이콘 색상 변경
        if (partIcon != null)
        {
            partIcon.color = GetColorForDamageLevel(bodyPart.damageLevel);
        }
    }
    
    private Color GetColorForDamageLevel(DamageLevel level)
    {
        return level switch
        {
            DamageLevel.None => healthyColor,
            DamageLevel.Minor => healthyColor,
            DamageLevel.Major => damagedColor,
            DamageLevel.Critical => criticalColor,
            DamageLevel.Destroyed => destroyedColor,
            _ => healthyColor
        };
    }
    
    /// <summary>
    /// 손상 단계 변경 이벤트 핸들러
    /// </summary>
    private void OnDamageLevelChanged(BodyPart part, DamageLevel level)
    {
        if (part == bodyPart)
        {
            UpdateStatus();
            ShowDamageEffect(level);
        }
    }
    
    /// <summary>
    /// 부위 파괴 이벤트 핸들러
    /// </summary>
    private void OnPartDestroyed(BodyPart part)
    {
        if (part == bodyPart)
        {
            UpdateStatus();
            ShowDestroyedEffect();
        }
    }
    
    /// <summary>
    /// 피해 효과 표시
    /// </summary>
    private void ShowDamageEffect(DamageLevel level)
    {
        // 피해 레벨에 따른 시각적 효과
        switch (level)
        {
            case DamageLevel.Minor:
                // 약간 깜빡이는 효과
                StartCoroutine(BlinkEffect(damagedColor, 0.5f));
                break;
            case DamageLevel.Major:
                // 더 강한 깜빡이는 효과
                StartCoroutine(BlinkEffect(damagedColor, 1f));
                break;
            case DamageLevel.Critical:
                // 빨간색으로 깜빡이는 효과
                StartCoroutine(BlinkEffect(criticalColor, 1.5f));
                break;
        }
    }
    
    /// <summary>
    /// 파괴 효과 표시
    /// </summary>
    private void ShowDestroyedEffect()
    {
        // 파괴 시 시각적 효과
        StartCoroutine(FadeToGray());
    }
    
    /// <summary>
    /// 깜빡이는 효과 코루틴
    /// </summary>
    private System.Collections.IEnumerator BlinkEffect(Color effectColor, float duration)
    {
        float elapsed = 0f;
        Color originalColor = partIcon.color;
        
        while (elapsed < duration)
        {
            float t = Mathf.PingPong(elapsed * 4f, 1f);
            partIcon.color = Color.Lerp(originalColor, effectColor, t);
            
            elapsed += Time.deltaTime;
            yield return null;
        }
        
        // 원래 색상으로 복구
        partIcon.color = GetColorForDamageLevel(bodyPart.damageLevel);
    }
    
    /// <summary>
    /// 회색으로 페이드하는 효과 코루틴
    /// </summary>
    private System.Collections.IEnumerator FadeToGray()
    {
        float elapsed = 0f;
        float duration = 1f;
        Color startColor = partIcon.color;
        
        while (elapsed < duration)
        {
            float t = elapsed / duration;
            partIcon.color = Color.Lerp(startColor, destroyedColor, t);
            
            elapsed += Time.deltaTime;
            yield return null;
        }
        
        partIcon.color = destroyedColor;
    }
    
    /// <summary>
    /// 부위 클릭 시 상세 정보 표시
    /// </summary>
    public void OnPartClicked()
    {
        if (bodyPart == null) return;
        
        ShowPartDetails();
    }
    
    private void ShowPartDetails()
    {
        string details = $"{bodyPart.partName} 상세 정보\n";
        details += $"HP: {bodyPart.currentHP}/{bodyPart.maxHP}\n";
        details += $"상태: {GetStatusText(bodyPart.damageLevel)}\n";
        
        if (bodyPart.isDestroyed)
        {
            var effect = bodyPart.GetDestroyEffect();
            details += $"파괴 효과: {effect.description}";
        }
        
        Debug.Log(details);
    }
    
    private string GetStatusText(DamageLevel level)
    {
        return level switch
        {
            DamageLevel.None => "정상",
            DamageLevel.Minor => "경미한 손상",
            DamageLevel.Major => "심각한 손상",
            DamageLevel.Critical => "위험한 손상",
            DamageLevel.Destroyed => "파괴됨",
            _ => "알 수 없음"
        };
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        BodyPart.OnDamageLevelChanged -= OnDamageLevelChanged;
        BodyPart.OnPartDestroyed -= OnPartDestroyed;
    }
}
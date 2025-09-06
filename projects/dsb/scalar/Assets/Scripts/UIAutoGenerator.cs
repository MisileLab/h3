using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class UIAutoGenerator : MonoBehaviour
{
    public Font defaultFont;
    
    // UI 생성 결과 저장
    private Canvas battleCanvas;
    private BattleUI battleUI;
    
    /// <summary>
    /// 전투 UI를 자동으로 생성합니다
    /// </summary>
    public BattleUI CreateBattleUI()
    {
        Debug.Log("=== UI 자동 생성 시작 ===");
        
        // 기존 UI 정리
        ClearExistingUI();
        
        // 1. Canvas 생성
        CreateCanvas();
        
        // 2. BattleUI 컴포넌트 생성
        CreateBattleUIComponent();
        
        // 3. UI 패널들 생성
        CreateUIElements();
        
        Debug.Log("=== UI 자동 생성 완료 ===");
        
        // 생성된 UI 구조 검증
        ValidateGeneratedUI();
        
        return battleUI;
    }
    
    /// <summary>
    /// 생성된 UI가 올바른지 검증합니다
    /// </summary>
    private void ValidateGeneratedUI()
    {
        if (battleUI == null)
        {
            Debug.LogError("BattleUI 컴포넌트가 생성되지 않았습니다!");
            return;
        }
        
        // 필수 UI 요소들 검증
        int missingCount = 0;
        if (battleUI.battlePanel == null) { Debug.LogWarning("battlePanel이 null입니다!"); missingCount++; }
        if (battleUI.actionPanel == null) { Debug.LogWarning("actionPanel이 null입니다!"); missingCount++; }
        if (battleUI.attackButton == null) { Debug.LogWarning("attackButton이 null입니다!"); missingCount++; }
        if (battleUI.apText == null) { Debug.LogWarning("apText가 null입니다!"); missingCount++; }
        if (battleUI.apSlider == null) { Debug.LogWarning("apSlider가 null입니다!"); missingCount++; }
        
        if (missingCount == 0)
        {
            Debug.Log("✅ UI 검증 완료 - 모든 필수 요소가 정상 생성됨");
        }
        else
        {
            Debug.LogWarning($"⚠️ UI 검증 결과 - {missingCount}개 요소가 누락됨");
        }
    }
    
    /// <summary>
    /// 기존 UI 요소들을 정리합니다
    /// </summary>
    private void ClearExistingUI()
    {
        // 기존 BattleUI 제거
        BattleUI[] existingUIs = FindObjectsOfType<BattleUI>();
        foreach (BattleUI ui in existingUIs)
        {
            if (ui != null)
            {
                Debug.Log($"기존 UI 제거: {ui.gameObject.name}");
                DestroyImmediate(ui.gameObject);
            }
        }
        
        // 기존 Canvas 제거 (UI 전용)
        Canvas[] existingCanvases = FindObjectsOfType<Canvas>();
        foreach (Canvas canvas in existingCanvases)
        {
            if (canvas != null && canvas.name.Contains("Battle"))
            {
                Debug.Log($"기존 Canvas 제거: {canvas.gameObject.name}");
                DestroyImmediate(canvas.gameObject);
            }
        }
    }
    
    /// <summary>
    /// Canvas를 생성합니다
    /// </summary>
    private void CreateCanvas()
    {
        GameObject canvasObj = new GameObject("BattleCanvas");
        battleCanvas = canvasObj.AddComponent<Canvas>();
        battleCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
        battleCanvas.sortingOrder = 100;
        
        // CanvasScaler 추가
        CanvasScaler canvasScaler = canvasObj.AddComponent<CanvasScaler>();
        canvasScaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        canvasScaler.referenceResolution = new Vector2(1920, 1080);
        canvasScaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        canvasScaler.matchWidthOrHeight = 0.5f;
        
        // GraphicRaycaster 추가
        canvasObj.AddComponent<GraphicRaycaster>();
        
        // EventSystem이 없으면 생성
        CreateEventSystemIfNeeded();
        
        Debug.Log("Canvas 생성 완료");
    }
    
    /// <summary>
    /// BattleUI 컴포넌트를 생성하고 설정합니다
    /// </summary>
    private void CreateBattleUIComponent()
    {
        GameObject battleUIObj = new GameObject("BattleUI");
        battleUIObj.transform.SetParent(battleCanvas.transform, false);
        
        battleUI = battleUIObj.AddComponent<BattleUI>();
        
        Debug.Log("BattleUI 컴포넌트 생성 완료");
    }
    
    /// <summary>
    /// UI 요소들을 생성합니다
    /// </summary>
    private void CreateUIElements()
    {
        // 1. 메인 전투 패널 생성
        GameObject battlePanel = CreatePanel("BattlePanel", battleCanvas.transform, new Vector2(1920, 1080));
        battleUI.battlePanel = battlePanel;
        
        // 2. 액션 패널 생성 (하단)
        GameObject actionPanel = CreatePanel("ActionPanel", battlePanel.transform, new Vector2(1200, 150));
        SetAnchor(actionPanel.GetComponent<RectTransform>(), AnchorPresets.BottomCenter);
        actionPanel.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, 75);
        battleUI.actionPanel = actionPanel;
        
        // 3. 상태 패널 생성 (좌측)
        GameObject statusPanel = CreatePanel("StatusPanel", battlePanel.transform, new Vector2(300, 600));
        SetAnchor(statusPanel.GetComponent<RectTransform>(), AnchorPresets.MiddleLeft);
        statusPanel.GetComponent<RectTransform>().anchoredPosition = new Vector2(150, 0);
        battleUI.statusPanel = statusPanel;
        
        // 4. 턴 정보 패널 생성 (상단)
        GameObject turnPanel = CreatePanel("TurnPanel", battlePanel.transform, new Vector2(800, 80));
        SetAnchor(turnPanel.GetComponent<RectTransform>(), AnchorPresets.TopCenter);
        turnPanel.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, -40);
        
        // 5. 대화 패널 생성 (중앙 하단)
        GameObject dialoguePanel = CreatePanel("DialoguePanel", battlePanel.transform, new Vector2(800, 120));
        SetAnchor(dialoguePanel.GetComponent<RectTransform>(), AnchorPresets.BottomCenter);
        dialoguePanel.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, 200);
        battleUI.dialoguePanel = dialoguePanel;
        dialoguePanel.SetActive(false);
        
        // 6. 협력 패널 생성
        GameObject cooperationPanel = CreatePanel("CooperationPanel", battlePanel.transform, new Vector2(600, 400));
        SetAnchor(cooperationPanel.GetComponent<RectTransform>(), AnchorPresets.MiddleCenter);
        battleUI.cooperationPanel = cooperationPanel;
        cooperationPanel.SetActive(false);
        
        // 7. UI 요소들 생성
        CreateActionButtons(actionPanel);
        CreateTurnInfo(turnPanel, battleUI);
        CreateDialogueElements(dialoguePanel, battleUI);
        CreateStatusElements(statusPanel, battleUI);
        
        // 초기 상태 설정 - UI를 즉시 볼 수 있도록 활성화
        battlePanel.SetActive(true);   // 전투 패널 활성화
        actionPanel.SetActive(true);   // 액션 패널 활성화 (테스트용)
        
        Debug.Log("UI 요소들 생성 완료");
    }
    
    // UI 생성 헬퍼 메서드들
    private enum AnchorPresets { TopCenter, BottomCenter, MiddleLeft, MiddleCenter }
    
    private GameObject CreatePanel(string name, Transform parent, Vector2 size)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent, false);
        
        // RectTransform 명시적 추가
        RectTransform rectTransform = panel.AddComponent<RectTransform>();
        rectTransform.sizeDelta = size;
        
        // Image 컴포넌트 추가 (UI 패널의 배경)
        Image image = panel.AddComponent<Image>();
        image.color = new Color(0, 0, 0, 0.3f);
        
        return panel;
    }
    
    private void SetAnchor(RectTransform rectTransform, AnchorPresets preset)
    {
        switch (preset)
        {
            case AnchorPresets.TopCenter:
                rectTransform.anchorMin = new Vector2(0.5f, 1f);
                rectTransform.anchorMax = new Vector2(0.5f, 1f);
                rectTransform.pivot = new Vector2(0.5f, 1f);
                break;
            case AnchorPresets.BottomCenter:
                rectTransform.anchorMin = new Vector2(0.5f, 0f);
                rectTransform.anchorMax = new Vector2(0.5f, 0f);
                rectTransform.pivot = new Vector2(0.5f, 0f);
                break;
            case AnchorPresets.MiddleLeft:
                rectTransform.anchorMin = new Vector2(0f, 0.5f);
                rectTransform.anchorMax = new Vector2(0f, 0.5f);
                rectTransform.pivot = new Vector2(0f, 0.5f);
                break;
            case AnchorPresets.MiddleCenter:
                rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
                rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
                rectTransform.pivot = new Vector2(0.5f, 0.5f);
                break;
        }
    }
    
    private GameObject CreateButton(string text, Transform parent, Vector2 size)
    {
        GameObject buttonObj = new GameObject(text + "Button");
        buttonObj.transform.SetParent(parent, false);
        
        // 버튼에 RectTransform 명시적 추가
        RectTransform rectTransform = buttonObj.AddComponent<RectTransform>();
        rectTransform.sizeDelta = size;
        
        // 버튼 Image와 Button 컴포넌트 추가
        Image image = buttonObj.AddComponent<Image>();
        image.color = new Color(0.2f, 0.3f, 0.8f, 0.8f);
        Button button = buttonObj.AddComponent<Button>();
        button.targetGraphic = image;
        
        // 버튼 텍스트 생성
        GameObject textObj = new GameObject("Text");
        textObj.transform.SetParent(buttonObj.transform, false);
        
        // 텍스트에 RectTransform 명시적 추가
        RectTransform textRect = textObj.AddComponent<RectTransform>();
        Text textComponent = textObj.AddComponent<Text>();
        textComponent.text = text;
        textComponent.font = defaultFont != null ? defaultFont : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        textComponent.fontSize = 16;
        textComponent.color = Color.white;
        textComponent.alignment = TextAnchor.MiddleCenter;
        
        // 텍스트를 버튼 전체에 맞춤
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        return buttonObj;
    }
    
    private void CreateActionButtons(GameObject actionPanel)
    {
        string[] buttonNames = {"공격", "방어", "스킬", "협력", "후퇴", "턴종료"};
        float buttonWidth = 180f, buttonHeight = 60f, spacing = 20f;
        float startX = -((buttonNames.Length - 1) * (buttonWidth + spacing)) / 2f;
        
        Button[] buttons = new Button[buttonNames.Length];
        for (int i = 0; i < buttonNames.Length; i++)
        {
            GameObject buttonObj = CreateButton(buttonNames[i], actionPanel.transform, new Vector2(buttonWidth, buttonHeight));
            buttonObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(startX + i * (buttonWidth + spacing), 0);
            buttons[i] = buttonObj.GetComponent<Button>();
        }
        
        battleUI.attackButton = buttons[0];
        battleUI.defendButton = buttons[1];
        battleUI.skillButton = buttons[2];
        battleUI.cooperationButton = buttons[3];
        battleUI.retreatButton = buttons[4];
        battleUI.endTurnButton = buttons[5];
    }
    
    private void CreateTurnInfo(GameObject turnPanel, BattleUI ui)
    {
        // 턴 정보 텍스트
        GameObject turnInfoObj = CreateText("TurnInfo", turnPanel.transform, "턴 1", 24);
        turnInfoObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(-300, 0);
        ui.turnInfoText = turnInfoObj.GetComponent<Text>();
        
        // 현재 액터 텍스트
        GameObject currentActorObj = CreateText("CurrentActor", turnPanel.transform, "현재 행동: ", 20);
        currentActorObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(-50, 0);
        ui.currentActorText = currentActorObj.GetComponent<Text>();
        
        // AP 표시 텍스트
        GameObject apTextObj = CreateText("APText", turnPanel.transform, "AP: 0/0", 18);
        apTextObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(200, 0);
        ui.apText = apTextObj.GetComponent<Text>();
        
        // AP 슬라이더 (간단한 버전)
        GameObject apSliderObj = CreateSimpleSlider("APSlider", turnPanel.transform, new Vector2(150, 20));
        apSliderObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(320, 0);
        ui.apSlider = apSliderObj.GetComponent<Slider>();
        
        Debug.Log("턴 정보 UI 생성 완료 (AP 표시 포함)");
    }
    
    private void CreateDialogueElements(GameObject dialoguePanel, BattleUI ui)
    {
        // 화자 이름
        GameObject speakerObj = CreateText("Speaker", dialoguePanel.transform, "화자", 18);
        speakerObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, 30);
        ui.speakerName = speakerObj.GetComponent<Text>();
        
        // 대화 텍스트
        GameObject dialogueObj = CreateText("DialogueText", dialoguePanel.transform, "대화 내용", 16);
        dialogueObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, -10);
        ui.dialogueText = dialogueObj.GetComponent<Text>();
        
        Debug.Log("대화 UI 생성 완료");
    }
    
    private void CreateStatusElements(GameObject statusPanel, BattleUI ui)
    {
        // 기계 상태 컨테이너
        GameObject containerObj = new GameObject("MechStatusContainer");
        containerObj.transform.SetParent(statusPanel.transform, false);
        
        // 컨테이너에 RectTransform 명시적 추가 (UI 레이아웃을 위해)
        RectTransform containerRect = containerObj.AddComponent<RectTransform>();
        containerRect.anchorMin = Vector2.zero;
        containerRect.anchorMax = Vector2.one;
        containerRect.offsetMin = Vector2.zero;
        containerRect.offsetMax = Vector2.zero;
        
        ui.mechStatusContainer = containerObj.transform;
        
        Debug.Log("상태 UI 생성 완료");
    }
    
    private GameObject CreateText(string name, Transform parent, string text, int fontSize)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.SetParent(parent, false);
        
        // 텍스트에 RectTransform 명시적 추가
        RectTransform rectTransform = textObj.AddComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(200, 30);
        
        // Text 컴포넌트 추가 및 설정
        Text textComponent = textObj.AddComponent<Text>();
        textComponent.text = text;
        textComponent.font = defaultFont != null ? defaultFont : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        textComponent.fontSize = fontSize;
        textComponent.color = Color.white;
        textComponent.alignment = TextAnchor.MiddleCenter;
        
        return textObj;
    }
    
    /// <summary>
    /// 간단한 슬라이더를 생성하는 헬퍼 메서드
    /// </summary>
    private GameObject CreateSimpleSlider(string name, Transform parent, Vector2 size)
    {
        GameObject sliderObj = new GameObject(name);
        sliderObj.transform.SetParent(parent, false);
        
        RectTransform sliderRect = sliderObj.AddComponent<RectTransform>();
        sliderRect.sizeDelta = size;
        
        Slider slider = sliderObj.AddComponent<Slider>();
        
        // Background
        GameObject background = new GameObject("Background");
        background.transform.SetParent(sliderObj.transform, false);
        // UI 요소에는 RectTransform이 자동으로 추가되지만, 명시적으로 Image 추가 시 확실히 하기 위해
        RectTransform backgroundRect = background.AddComponent<RectTransform>();
        Image backgroundImage = background.AddComponent<Image>();
        backgroundImage.color = new Color(0.2f, 0.2f, 0.2f, 0.8f);
        
        backgroundRect.anchorMin = Vector2.zero;
        backgroundRect.anchorMax = Vector2.one;
        backgroundRect.offsetMin = Vector2.zero;
        backgroundRect.offsetMax = Vector2.zero;
        
        // Fill Area
        GameObject fillArea = new GameObject("Fill Area");
        fillArea.transform.SetParent(sliderObj.transform, false);
        // Fill Area에 RectTransform 명시적으로 추가
        RectTransform fillAreaRect = fillArea.AddComponent<RectTransform>();
        fillAreaRect.anchorMin = Vector2.zero;
        fillAreaRect.anchorMax = Vector2.one;
        fillAreaRect.offsetMin = Vector2.zero;
        fillAreaRect.offsetMax = Vector2.zero;
        
        // Fill
        GameObject fill = new GameObject("Fill");
        fill.transform.SetParent(fillArea.transform, false);
        // Fill에 RectTransform 명시적으로 추가
        RectTransform fillRect = fill.AddComponent<RectTransform>();
        Image fillImage = fill.AddComponent<Image>();
        fillImage.color = new Color(0.2f, 0.6f, 1f, 0.8f); // 파란색
        
        fillRect.anchorMin = Vector2.zero;
        fillRect.anchorMax = Vector2.one;
        fillRect.offsetMin = Vector2.zero;
        fillRect.offsetMax = Vector2.zero;
        
        slider.fillRect = fillRect;
        slider.value = 0.5f;
        slider.minValue = 0f;
        slider.maxValue = 1f;
        
        return sliderObj;
    }
    
    /// <summary>
    /// EventSystem이 없으면 생성합니다 (UI 상호작용을 위해 필요)
    /// </summary>
    private void CreateEventSystemIfNeeded()
    {
        // 기존 EventSystem이 있는지 확인
        EventSystem existingEventSystem = FindObjectOfType<EventSystem>();
        if (existingEventSystem == null)
        {
            // EventSystem GameObject 생성
            GameObject eventSystemObj = new GameObject("EventSystem");
            EventSystem eventSystem = eventSystemObj.AddComponent<EventSystem>();
            
            // Input System 패키지 호환성 체크
            if (TryAddInputSystemUIModule(eventSystemObj))
            {
                Debug.Log("EventSystem 생성 완료 - Input System UI Module 사용");
            }
            else
            {
                // Input System이 없거나 사용할 수 없으면 StandaloneInputModule 사용
                eventSystemObj.AddComponent<StandaloneInputModule>();
                Debug.Log("EventSystem 생성 완료 - Standalone Input Module 사용");
            }
        }
        else
        {
            Debug.Log("기존 EventSystem 발견 - 버튼 클릭이 작동할 것입니다.");
            
            // 기존 EventSystem의 Input Module 확인 및 수정
            FixExistingEventSystemInputModule(existingEventSystem);
        }
    }
    
    /// <summary>
    /// Input System UI Module을 추가하려고 시도합니다
    /// </summary>
    private bool TryAddInputSystemUIModule(GameObject eventSystemObj)
    {
        try
        {
            // Reflection을 사용해서 InputSystemUIInputModule이 있는지 확인
            System.Type inputSystemUIModuleType = System.Type.GetType("UnityEngine.InputSystem.UI.InputSystemUIInputModule, Unity.InputSystem");
            
            if (inputSystemUIModuleType != null)
            {
                // InputSystemUIInputModule 추가
                eventSystemObj.AddComponent(inputSystemUIModuleType);
                return true;
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"InputSystemUIInputModule 추가 실패: {ex.Message}");
        }
        
        return false;
    }
    
    /// <summary>
    /// 기존 EventSystem의 Input Module을 수정합니다
    /// </summary>
    private void FixExistingEventSystemInputModule(EventSystem eventSystem)
    {
        // StandaloneInputModule이 있으면 제거하고 InputSystemUIInputModule로 교체
        StandaloneInputModule standaloneModule = eventSystem.GetComponent<StandaloneInputModule>();
        if (standaloneModule != null)
        {
            if (TryAddInputSystemUIModule(eventSystem.gameObject))
            {
                DestroyImmediate(standaloneModule);
                Debug.Log("기존 EventSystem을 Input System 호환 모드로 업데이트했습니다.");
            }
            else
            {
                Debug.LogWarning("Input System UI Module을 추가할 수 없어 StandaloneInputModule을 유지합니다. Player Settings에서 Input Handling을 'Both' 또는 'Input Manager (Old)'로 변경해보세요.");
            }
        }
    }
}

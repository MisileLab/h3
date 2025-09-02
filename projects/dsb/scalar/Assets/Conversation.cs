using UnityEngine;
using Ink.Runtime;
using UnityEngine.UI;
using TMPro;

public class Conversation : MonoBehaviour
{
    [SerializeField]
    private TextAsset _inkJsonAsset;
    private Story _story;

    [SerializeField]
    private TextMeshProUGUI _textField;

    [SerializeField]
    private GameObject _choiceButtonContainerPrefab;

    private VerticalLayoutGroup _choiceButtonContainer;

    [SerializeField]
    private Button _choiceButtonTemplate;

    [SerializeField]
    private Color _normalTextColor;

    [SerializeField]
    private Color _thoughtTextColor;

    void Start()
    {
        if (_choiceButtonContainerPrefab != null)
        {
            var containerObject = Instantiate(_choiceButtonContainerPrefab, transform);
            _choiceButtonContainer = containerObject.GetComponent<VerticalLayoutGroup>();
        }
        else
        {
            Debug.LogError("ChoiceButtonContainerPrefab is not set in the inspector.");
        }
        StartStory();
    }

    

    private void StartStory()
    {
        if (_inkJsonAsset == null) {
            Debug.LogError("Ink JSON Asset is not assigned.");
            return;
        }
        _story = new Story(_inkJsonAsset.text);
        Debug.Log("Story started.");
        DisplayNextLine();
    }
    
    public void DisplayNextLine()
    {
        Debug.Log("DisplayNextLine called.");
        if (_story.canContinue)
        {
            string text = _story.Continue(); // gets next line
            text = text?.Trim(); // removes white space from text
            Debug.Log("Story Text: " + text);
            ApplyStyling();

            if (_textField != null) {
                _textField.text = text; // displays new text
            } else {
                Debug.LogError("Text field is not assigned.");
            }
        }
        else if (_story.currentChoices.Count > 0)
        {
            DisplayChoices();
        }
        else
        {
            Debug.Log("Story cannot continue.");
        }
    }

    private void DisplayChoices()
    {
        // checks if choices are already being displayed
        if (_choiceButtonContainer != null && _choiceButtonContainer.transform.childCount > 0) return;

        for (int i = 0; i < _story.currentChoices.Count; i++) // iterates through all choices
        {
            var choice = _story.currentChoices[i];
            var button = CreateChoiceButton(choice.text); // creates a choice button by cloning

            button.onClick.AddListener(() => OnClickChoiceButton(choice));
        }
    }

    Button CreateChoiceButton(string text)
    {
        // clones the button template
        var choiceButton = Instantiate(_choiceButtonTemplate);
        
        if (_choiceButtonContainer != null)
        {
            choiceButton.transform.SetParent(_choiceButtonContainer.transform, false);
        }
        
        // sets text on the button
        var buttonText = choiceButton.GetComponentInChildren<TextMeshProUGUI>();
        if (buttonText != null)
        {
            buttonText.text = text;
            
            // Configure text for auto-sizing
            buttonText.enableAutoSizing = true;
            buttonText.fontSizeMin = 12f;
            buttonText.fontSizeMax = 24f;
            buttonText.alignment = TextAlignmentOptions.Center;
            
            // Add padding around text for better appearance
            buttonText.margin = new Vector4(10f, 5f, 10f, 5f); // left, top, right, bottom padding
        }
        
        // Add ContentSizeFitter to auto-adjust button size based on text
        ContentSizeFitter buttonSizeFitter = choiceButton.GetComponent<ContentSizeFitter>();
        if (buttonSizeFitter == null)
        {
            buttonSizeFitter = choiceButton.gameObject.AddComponent<ContentSizeFitter>();
        }
        buttonSizeFitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        buttonSizeFitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        
        // Add LayoutElement to control min/max sizes and centering
        LayoutElement layoutElement = choiceButton.GetComponent<LayoutElement>();
        if (layoutElement == null)
        {
            layoutElement = choiceButton.gameObject.AddComponent<LayoutElement>();
        }
        layoutElement.minWidth = 100f; // Minimum button width
        layoutElement.minHeight = 40f; // Minimum button height
        layoutElement.preferredWidth = -1f; // Let ContentSizeFitter control width
        layoutElement.preferredHeight = -1f; // Let ContentSizeFitter control height
        layoutElement.flexibleWidth = 0f; // Don't expand to fill container
        layoutElement.flexibleHeight = 0f; // Don't expand vertically

        // activate the cloned button
        choiceButton.gameObject.SetActive(true);

        return choiceButton;
    }

    void OnClickChoiceButton(Choice choice)
    {
        _story.ChooseChoiceIndex(choice.index); // tells ink which choice was selected
        RefreshChoiceView(); // removes choices from the screen
        DisplayNextLine();
    }

    void RefreshChoiceView()
    {
        if (_choiceButtonContainer != null)
        {
            foreach (Transform child in _choiceButtonContainer.transform)
            {
                Destroy(child.gameObject);
            }
        }
    }

    private void ApplyStyling()
    {
        if (_story.currentTags.Contains("thought"))
        {
            _textField.color = _thoughtTextColor;
            _textField.fontStyle = FontStyles.Italic;
        }
        else
        {
            _textField.color = _normalTextColor;
            _textField.fontStyle = FontStyles.Normal;
        }
    }
}

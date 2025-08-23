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

    //[SerializeField]
    //private VerticalLayoutGroup _choiceButtonContainer;

    //[SerializeField]
    //private Button _choiceButtonPrefab;

    [SerializeField]
    private Color _normalTextColor;

    [SerializeField]
    private Color _thoughtTextColor;

    void Start()
    {
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
        else
        {
            Debug.Log("Story cannot continue.");
        }
        //else if (_story.currentChoices.Count > 0)
        //{
        //    DisplayChoices();
        //}
    }

    /*
    private void DisplayChoices()
    {
        // checks if choices are already being displaye
        if (_choiceButtonContainer.GetComponentsInChildren<Button>().Length > 0) return;

        for (int i = 0; i < _story.currentChoices.Count; i++) // iterates through all choices
        {

            var choice = _story.currentChoices[i];
            var button = CreateChoiceButton(choice.text); // creates a choice button

            button.onClick.AddListener(() => OnClickChoiceButton(choice));
        }
    }

    Button CreateChoiceButton(string text)
    {
        // creates the button from a prefab
        var choiceButton = Instantiate(_choiceButtonPrefab);
        choiceButton.transform.SetParent(_choiceButtonContainer.transform, false);
        
        // sets text on the button
        var buttonText = choiceButton.GetComponentInChildren<TextMeshProUGUI>();
        buttonText.text = text;

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
            foreach (var button in _choiceButtonContainer.GetComponentsInChildren<Button>())
            {
                Destroy(button.gameObject);
            }
        }
    }
    */

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

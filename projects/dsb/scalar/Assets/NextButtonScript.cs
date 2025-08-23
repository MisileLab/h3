using UnityEngine;

public class NextButtonScript : MonoBehaviour
{
  private Conversation _conversation;

  void Start()
  {
    _conversation = FindFirstObjectByType<Conversation>();

    if (_conversation == null)
    {
      Debug.LogError("Conversation script was not found!");
    }
  }

  public void OnClick()
  {
    _conversation?.DisplayNextLine();
  }
}

using UnityEngine;

public class GridManager : MonoBehaviour
{
  [SerializeField] private int _width, _height;
  [SerializeField] private GameObject _tilePrefab;

  void Start()
  {
    _tilePrefab.GetComponent<Renderer>().enabled = false;
    Camera.main.transform.position = new Vector3(_width / 2f - 0.5f, _height / 2f - 0.5f);
    generateGrid();
  }

  void generateGrid()
  {
    // Destroy old child tiles
    for (int i = transform.childCount - 1; i >= 0; i--)
    {
      Destroy(transform.GetChild(i).gameObject);
    }

    // Instantiate visible copies
    for (int x = 0; x < _width; x++)
    {
      for (int y = 0; y < _height; y++)
      {
        GameObject tile = Instantiate(_tilePrefab, new Vector3(x, y, 0), Quaternion.identity);
        tile.name = $"{x}-{y}";
        tile.transform.SetParent(transform);
        tile.GetComponent<Renderer>().enabled = true;
      }
    }
  }
}

using UnityEngine;
using UnityEngine.Tilemaps;

public class TileMapScript: MonoBehaviour
{
  private Tilemap tilemap;
  private TileBase tile;

  void Start() {
    tilemap = gameObject.GetComponent<Tilemap>();
    Debug.Assert(tilemap != null);
  }

  // Update is called once per frame
  void Update()
  {
    if (Input.GetMouseButtonDown(0))
    {
      Vector3Int tilemapPos = tilemap.WorldToCell(Camera.main.ScreenToWorldPoint(Input.mousePosition));
      tilemapPos.z = 0;
      TileBase tile = tilemap.GetTile(tilemapPos);
      if (tile)
      {
        Debug.Log("Tile is " + tile.name);
      }
    }
  }
}

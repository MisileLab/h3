using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

public enum Material {
  material
}

public enum ItemType {
  material,
  researcher,
  reformer,
  extractor
}

public class Item {
  public ItemType type;
  public int amount;
}

public class Tile {
  public Material material;
  public int amount;
  public int x;
  public int y;
}

public class TileMapScript: MonoBehaviour
{
  private Tilemap tilemap;
  private Dictionary<Vector3Int, Tile> tiles = new();
  private List<Item> items = new();
  private Vector3Int pastTilePos;

  // tiles
  public TileBase normalTile;
  public TileBase hoveredTile;
  public TileBase researchedTile;
  public TileBase emptyTile;

  bool IsSamePos(Vector3Int a, Vector3Int b) {
    return a.x == b.x && a.y == b.y;
  }

  void Render() {
    foreach (Vector3Int pos in tilemap.cellBounds.allPositionsWithin) {
      if (tiles.ContainsKey(pos)) {
        if (tiles[pos].amount == 0) {
          tilemap.SetTile(pos, emptyTile);
        } else {
          tilemap.SetTile(pos, researchedTile);
        }
      }
    }
  }

  void Start() {
    tilemap = gameObject.GetComponent<Tilemap>();
    Debug.Assert(tilemap != null);
    Render();
  }

  // Update is called once per frame
  void Update() {
    Vector3Int tilemapPos = tilemap.WorldToCell(Camera.main.ScreenToWorldPoint(Input.mousePosition));
    tilemapPos.z = 0;
    TileBase tile = tilemap.GetTile(tilemapPos);
    if (tile) {
      if (Input.GetMouseButtonDown(0)) {
        Debug.Log("clicked");
      } else if (!IsSamePos(tilemapPos, pastTilePos)) {
        tilemap.SetTile(tilemapPos, hoveredTile);
        if (pastTilePos != null) {
          tilemap.SetTile(pastTilePos, normalTile);
        }
        Debug.Log(tilemapPos + " " + pastTilePos);
        pastTilePos = tilemapPos;
      }
    }
  }
}

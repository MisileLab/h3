using System.Collections.Generic;
using System.Runtime.Serialization.Formatters;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Tilemaps;

public enum TileType {
  // Non machine
  Empty,
  Wire,

  // machine
  Producer,
  Launcher
}

public class Tile {
  public TileType type;
  public int x;
  public int y;
}

public class TileMapScript: MonoBehaviour
{
  private Tilemap tilemap;
  private Dictionary<Vector3Int, Tile> tiles = new();
  private Vector3Int pastTilePos;

  // tiles
  public TileBase normalTile;
  public TileBase hoveredTile;

  bool isSamePos(Vector3Int a, Vector3Int b) {
    return a.x == b.x && a.y == b.y;
  }

  void render() {
    foreach (Vector3Int pos in tilemap.cellBounds.allPositionsWithin) {
      Debug.Assert(tilemap.HasTile(pos));
      tiles[pos] = new Tile { type = TileType.Empty, x = pos.x, y = pos.y };
    }
  }

  void Start() {
    tilemap = gameObject.GetComponent<Tilemap>();
    Debug.Assert(tilemap != null);
    render();
  }

  // Update is called once per frame
  void Update() {
    Vector3Int tilemapPos = tilemap.WorldToCell(Camera.main.ScreenToWorldPoint(Input.mousePosition));
    tilemapPos.z = 0;
    TileBase tile = tilemap.GetTile(tilemapPos);
    if (tile) {
      if (Input.GetMouseButtonDown(0)) {
        Debug.Log("clicked");
      } else if (!isSamePos(tilemapPos, pastTilePos)) {
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

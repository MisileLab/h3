using System;
using System.Collections.Generic;
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

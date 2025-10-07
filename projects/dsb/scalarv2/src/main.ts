import * as PIXI from "pixi.js";
import {
  storyManager,
  EPISODE_1,
  DATA_FRAGMENTS,
  FREE_EXPLORATION_SCENES,
} from "./story";

// Extend Window interface for our custom properties
declare global {
  interface Window {
    pendingNewGame?: boolean;
  }
}

// --- THEME & STYLING (from Pixi) ---
const THEME = {
  background: 0x1a1a2e,
  node: 0x4a4e69,
  nodeHover: 0x9a8c98,
  nodeActive: 0xf2e9e4,
  line: 0x4a4e69,
  lineActive: 0x9a8c98,
  text: 0xf2e9e4,
  accent: 0xe94560,
  resource: 0x22aadd,
  event: 0xfca311,
  combat: 0xd90429,
  escape: 0x52b788,
  hpGreen: 0x28a745,
  hpRed: 0xdc3545,
  heal: 0x90ee90,
  guard: 0xadd8e6,
  moveHighlight: 0xfca311,
  attackHighlight: 0xdc3545,
};

// --- GAME CONFIG ---
const MAP_COLS = 12;
const MAP_ROWS = 5;
const NODE_RADIUS = 25;
const NODE_VERTICAL_SPACING = 120;
const NODE_HORIZONTAL_SPACING = 150;

const COMBAT_GRID_COLS = 10;
const COMBAT_GRID_ROWS = 6;
const CELL_SIZE = 90;

// --- TYPE DEFINITIONS ---
type NodeType =
  | "START"
  | "RESOURCE"
  | "EVENT"
  | "COMBAT"
  | "ESCAPE"
  | "EMPTY"
  | "STORY";
type UnitRole = "Guardian" | "Supporter" | "Scout" | "Attacker" | "Enemy";
type PlayerAction = "move" | "attack" | "special";

interface GameNode {
  id: number;
  x: number;
  y: number;
  col: number;
  row: number;
  type: NodeType;
  connections: number[];
  graphics: PIXI.Graphics;
  text: PIXI.Text;
}

interface Unit {
  id: string;
  name: string;
  role: UnitRole;
  description: string;
  isPlayer: boolean;
  hp: number;
  maxHp: number;
  attack: number;
  range: number;
  speed: number;
  movement: number;
  gridX: number;
  gridY: number;
  isDefending: boolean;
  sprite: PIXI.Graphics;
  hpBar: PIXI.Graphics;
  nameTag: PIXI.Text;
  say(message: string): void;
}

interface Party {
  members: Unit[];
  currentNodeId: number;
  previousNodeId: number | null;
  mapSprite: PIXI.Graphics;
}

// Represents the state for a single mission
interface GameState {
  map: Map<number, GameNode>;
  party: Party;
  resources: {
    parts: number;
    data: number;
    energy: number;
  };
  isInCombat: boolean;
  combat: {
    enemies: Unit[];
    turnOrder: Unit[];
    turnIndex: number;
    grid: (Unit | null)[][];
    activePlayerAction: PlayerAction | null;
  };
  story: {
    currentEpisode: string | null;
    unlockedFragments: string[];
    storyFlags: Record<string, boolean>;
  };
}

// Represents the persistent state of the player across all missions
interface MetaState {
  totalParts: number;
  totalData: number;
  totalEnergy: number;
  availableUnits: Unit[];
}

// --- UI SELECTORS ---
const setupContainer = document.getElementById(
  "setup-container",
) as HTMLElement;
const gameBoard = document.getElementById("game-board") as HTMLElement;

const resPartsEl = document.getElementById("res-parts");
const resDataEl = document.getElementById("res-data");
const resEnergyEl = document.getElementById("res-energy");
const logContentEl = document.getElementById("log-content");
const partyStatusEl = document.querySelector("#party-status");

// --- STORY UI ELEMENTS ---
const dialogueBox = document.getElementById("dialogue-box");
const dialoguePortrait = document.getElementById("dialogue-portrait");
const dialogueName = document.getElementById("dialogue-name");
const dialogueText = document.getElementById("dialogue-text");
const dialogueChoices = document.getElementById("dialogue-choices");
const dataFragmentsEl = document.getElementById("data-fragments");
const fragmentsListEl = document.getElementById("fragments-list");

// --- MAP CONTROL ELEMENTS ---
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const zoomResetBtn = document.getElementById("zoom-reset");
const centerMapBtn = document.getElementById("center-map");
const saveGameBtn = document.getElementById("save-game");
const loadGameBtn = document.getElementById("load-game");
const newGameBtn = document.getElementById("new-game");

// --- PIXI APP & CONTAINERS ---
let app: PIXI.Application;
const mapContainer = new PIXI.Container();
const combatContainer = new PIXI.Container();
const gridContainer = new PIXI.Container();
const highlightContainer = new PIXI.Container();

// --- MAP SCROLLING ---
let isDragging = false;
const dragStart = { x: 0, y: 0 };
const mapScroll = { x: 0, y: 0 };
const ZOOM_SPEED = 0.1;
let currentZoom = 1;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2.0;

// --- GAME STATE FACTORIES & DATA ---
const createUnit = (
  id: string,
  name: string,
  role: UnitRole,
  description: string,
  isPlayer: boolean,
): Unit => ({
  id,
  name,
  role,
  description,
  isPlayer,
  hp: 1,
  maxHp: 1,
  attack: 1,
  range: 1,
  speed: 1,
  movement: 1,
  gridX: 0,
  gridY: 0,
  isDefending: false,
  sprite: new PIXI.Graphics(),
  hpBar: new PIXI.Graphics(),
  nameTag: new PIXI.Text(name, { fill: THEME.text, fontSize: 14 }),
  say(message: string) {
    logEvent(
      `<span style="color: ${this.isPlayer ? THEME.heal : THEME.accent};">${this.name}:</span> "${message}"`,
    );
  },
});

const ALL_POSSIBLE_UNITS = {
  rex: () => {
    const unit = createUnit(
      "rex",
      "Rex",
      "Guardian",
      "A sturdy machine that can protect others.",
      true,
    );
    unit.maxHp = 150;
    unit.hp = 150;
    unit.attack = 12;
    unit.range = 1;
    unit.speed = 30;
    unit.movement = 2;
    return unit;
  },
  luna: () => {
    const unit = createUnit(
      "luna",
      "Luna",
      "Supporter",
      "A nimble unit that can repair allies from a distance.",
      true,
    );
    unit.maxHp = 80;
    unit.hp = 80;
    unit.attack = 8;
    unit.range = 3;
    unit.speed = 50;
    unit.movement = 3;
    return unit;
  },
  zero: () => {
    const unit = createUnit(
      "zero",
      "Zero",
      "Attacker",
      "A high-damage attacker with moderate range.",
      true,
    );
    unit.maxHp = 100;
    unit.hp = 100;
    unit.attack = 20;
    unit.range = 2;
    unit.speed = 40;
    unit.movement = 3;
    return unit;
  },
  nova: () => {
    const unit = createUnit(
      "nova",
      "Nova",
      "Scout",
      "A fast-moving scout that can cover long distances.",
      true,
    );
    unit.maxHp = 70;
    unit.hp = 70;
    unit.attack = 10;
    unit.range = 1;
    unit.speed = 60;
    unit.movement = 4;
    return unit;
  },
};

// --- GAME STATE ---
let gameState: GameState;
let metaState: MetaState;

const defaultMetaState: () => MetaState = () => ({
  totalParts: 0,
  totalData: 0,
  totalEnergy: 0,
  availableUnits: [ALL_POSSIBLE_UNITS.rex(), ALL_POSSIBLE_UNITS.luna()],
});

function saveMetaState() {
  try {
    localStorage.setItem("scalar_metaState", JSON.stringify(metaState));
  } catch (e) {
    console.error("Failed to save state:", e);
  }
}

function loadMetaState() {
  try {
    const savedState = localStorage.getItem("scalar_metaState");
    if (savedState) {
      metaState = JSON.parse(savedState);
    } else {
      metaState = defaultMetaState();
    }
  } catch (e) {
    console.error("Failed to load state:", e);
    metaState = defaultMetaState();
  }
}

// --- GAME STATE SAVE/LOAD ---
function saveGameState() {
  if (!gameState) return;

  try {
    // Create a serializable version of gameState
    const serializableState = {
      map: Array.from(gameState.map.entries()).map(([, node]) => ({
        id: node.id,
        x: node.x,
        y: node.y,
        col: node.col,
        row: node.row,
        type: node.type,
        connections: node.connections,
        storyId: (node as GameNode & { storyId?: string }).storyId,
        specialData: (node as GameNode & { specialData?: unknown }).specialData,
      })),
      party: {
        members: gameState.party.members.map((member) => ({
          id: member.id,
          name: member.name,
          role: member.role,
          description: member.description,
          isPlayer: member.isPlayer,
          hp: member.hp,
          maxHp: member.maxHp,
          attack: member.attack,
          range: member.range,
          speed: member.speed,
          movement: member.movement,
          gridX: member.gridX,
          gridY: member.gridY,
          isDefending: member.isDefending,
        })),
        currentNodeId: gameState.party.currentNodeId,
        previousNodeId: gameState.party.previousNodeId,
      },
      resources: gameState.resources,
      isInCombat: gameState.isInCombat,
      combat: gameState.combat,
      story: gameState.story,
    };

    localStorage.setItem("scalar_gameState", JSON.stringify(serializableState));
    console.log("Game state saved successfully");
  } catch (e) {
    console.error("Failed to save game state:", e);
  }
}

function loadGameState(): boolean {
  try {
    const savedState = localStorage.getItem("scalar_gameState");
    if (!savedState) {
      console.log("No saved game state found");
      return false;
    }

    const serializableState = JSON.parse(savedState);

    // Reconstruct the map
    const reconstructedMap = new Map<number, GameNode>();
    serializableState.map.forEach(
      (nodeData: {
        id: number;
        x: number;
        y: number;
        col: number;
        row: number;
        type: NodeType;
        connections: number[];
        storyId?: string;
        specialData?: unknown;
      }) => {
        const node: GameNode = {
          id: nodeData.id,
          x: nodeData.x,
          y: nodeData.y,
          col: nodeData.col,
          row: nodeData.row,
          type: nodeData.type,
          connections: nodeData.connections,
          graphics: new PIXI.Graphics(),
          text: new PIXI.Text("", { fill: THEME.text, fontSize: 20 }),
        };

        // Store additional data
        if (nodeData.storyId) {
          (node as GameNode & { storyId?: string }).storyId = nodeData.storyId;
        }
        if (nodeData.specialData) {
          (node as GameNode & { specialData?: unknown }).specialData =
            nodeData.specialData;
        }

        reconstructedMap.set(node.id, node);
      },
    );

    // Reconstruct party members
    const reconstructedMembers = serializableState.party.members.map(
      (memberData: {
        id: string;
        name: string;
        role: UnitRole;
        description: string;
        isPlayer: boolean;
        hp: number;
        maxHp: number;
        attack: number;
        range: number;
        speed: number;
        movement: number;
        gridX: number;
        gridY: number;
        isDefending: boolean;
      }) => {
        const unit = createUnit(
          memberData.id,
          memberData.name,
          memberData.role,
          memberData.description,
          memberData.isPlayer,
        );

        unit.hp = memberData.hp;
        unit.maxHp = memberData.maxHp;
        unit.attack = memberData.attack;
        unit.range = memberData.range;
        unit.speed = memberData.speed;
        unit.movement = memberData.movement;
        unit.gridX = memberData.gridX;
        unit.gridY = memberData.gridY;
        unit.isDefending = memberData.isDefending;

        return unit;
      },
    );

    // Reconstruct game state
    gameState = {
      map: reconstructedMap,
      party: {
        members: reconstructedMembers,
        currentNodeId: serializableState.party.currentNodeId,
        previousNodeId: serializableState.party.previousNodeId,
        mapSprite: new PIXI.Graphics(),
      },
      resources: serializableState.resources,
      isInCombat: serializableState.isInCombat,
      combat: serializableState.combat,
      story: serializableState.story,
    };

    console.log("Game state loaded successfully");
    return true;
  } catch (e) {
    console.error("Failed to load game state:", e);
    return false;
  }
}

function hasSavedGame(): boolean {
  return localStorage.getItem("scalar_gameState") !== null;
}

// --- COMBAT UI (PIXI) ---
const attackButton = new PIXI.Graphics()
  .beginFill(THEME.accent)
  .drawRoundedRect(0, 0, 100, 50, 10)
  .endFill();
const specialButton = new PIXI.Graphics()
  .beginFill(THEME.resource)
  .drawRoundedRect(0, 0, 100, 50, 10)
  .endFill();
const moveButton = new PIXI.Graphics()
  .beginFill(THEME.guard)
  .drawRoundedRect(0, 0, 100, 50, 10)
  .endFill();
const fleeButton = new PIXI.Graphics()
  .beginFill(THEME.node)
  .drawRoundedRect(0, 0, 100, 50, 10)
  .endFill();

const setupButton = (btn: PIXI.Graphics, text: string, action: () => void) => {
  const btnText = new PIXI.Text(text, {
    fill: THEME.text,
    fontSize: 18,
    align: "center",
  });
  btn.addChild(btnText);
  btnText.anchor.set(0.5);
  btnText.position.set(50, 25);
  btn.eventMode = "static";
  btn.cursor = "pointer";
  btn.on("pointerdown", action);
  return btn;
};

setupButton(attackButton, "ATTACK", () => setPlayerAction("attack"));
setupButton(specialButton, "SPECIAL", () => setPlayerAction("special"));
setupButton(moveButton, "MOVE", () => setPlayerAction("move"));
setupButton(fleeButton, "FLEE", () => fleeCombat());

// --- MAP SCROLLING FUNCTIONS ---

function setupMapScrolling() {
  if (!app) return;

  // 맵 컨트롤 버튼 이벤트 리스너 설정
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      const newZoom = Math.min(MAX_ZOOM, currentZoom + ZOOM_SPEED);
      if (newZoom !== currentZoom) {
        currentZoom = newZoom;
        updateMapTransform();
      }
    });
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      const newZoom = Math.max(MIN_ZOOM, currentZoom - ZOOM_SPEED);
      if (newZoom !== currentZoom) {
        currentZoom = newZoom;
        updateMapTransform();
      }
    });
  }

  if (zoomResetBtn) {
    zoomResetBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      resetMapView();
    });
  }

  if (centerMapBtn) {
    centerMapBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      centerMapOnNode(gameState.party.currentNodeId);
    });
  }

  // Save/Load/New Game button event listeners
  if (saveGameBtn) {
    saveGameBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      saveGameState();
      logEvent("Game saved manually!");
      showDialogue("Crystal", "게임이 저장되었어요, 스칼라!", ["알겠어요"]);
    });
  }

  if (loadGameBtn) {
    loadGameBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      const loaded = loadGameState();
      if (loaded) {
        logEvent("Game loaded successfully!");
        updateUI();
        renderMap();
        showDialogue("Crystal", "게임을 불러왔어요! 이전부터 계속해요.", [
          "계속할게요",
        ]);
      } else {
        logEvent("No saved game found!");
        showDialogue("Crystal", "저장된 게임이 없네요...", ["알겠어요"]);
      }
    });
  }

  if (newGameBtn) {
    newGameBtn.addEventListener("click", () => {
      if (gameState.isInCombat) return;
      // Set a flag for new game confirmation
      window.pendingNewGame = true;
      showDialogue(
        "Crystal",
        "정말로 새 게임을 시작할까요? 현재 진행 상황은 사라져요.",
        ["새 게임 시작", "취소"],
      );
    });
  }

  // 마우스 휠로 줌 인/아웃
  app.view.addEventListener("wheel", (event: WheelEvent) => {
    if (gameState.isInCombat) return;

    event.preventDefault();

    const delta = event.deltaY > 0 ? -ZOOM_SPEED : ZOOM_SPEED;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, currentZoom + delta));

    if (newZoom !== currentZoom) {
      const mousePos = { x: event.clientX, y: event.clientY };

      const zoomFactor = newZoom / currentZoom;
      mapScroll.x = mousePos.x - (mousePos.x - mapScroll.x) * zoomFactor;
      mapScroll.y = mousePos.y - (mousePos.y - mapScroll.y) * zoomFactor;

      currentZoom = newZoom;
      updateMapTransform();
    }
  });

  // 마우스 드래그로 맵 이동
  app.view.addEventListener("mousedown", (event: MouseEvent) => {
    if (gameState.isInCombat) return;

    isDragging = true;
    dragStart.x = event.clientX - mapScroll.x;
    dragStart.y = event.clientY - mapScroll.y;
    app.view.style.cursor = "grabbing";
  });

  app.view.addEventListener("mousemove", (event: MouseEvent) => {
    if (gameState.isInCombat || !isDragging) return;

    mapScroll.x = event.clientX - dragStart.x;
    mapScroll.y = event.clientY - dragStart.y;
    updateMapTransform();
  });

  app.view.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      app.view.style.cursor = "default";
    }
  });

  app.view.addEventListener("mouseleave", () => {
    if (isDragging) {
      isDragging = false;
      app.view.style.cursor = "default";
    }
  });

  // 키보드 스크롤 (화살표 키)
  document.addEventListener("keydown", (event: KeyboardEvent) => {
    if (gameState.isInCombat) return;

    const scrollAmount = 50;
    switch (event.key) {
      case "ArrowLeft":
        mapScroll.x += scrollAmount;
        break;
      case "ArrowRight":
        mapScroll.x -= scrollAmount;
        break;
      case "ArrowUp":
        mapScroll.y += scrollAmount;
        break;
      case "ArrowDown":
        mapScroll.y -= scrollAmount;
        break;
      case "=":
      case "+": {
        const zoomIn = Math.min(MAX_ZOOM, currentZoom + ZOOM_SPEED);
        if (zoomIn !== currentZoom) {
          currentZoom = zoomIn;
          updateMapTransform();
        }
        break;
      }
      case "-": {
        const zoomOut = Math.max(MIN_ZOOM, currentZoom - ZOOM_SPEED);
        if (zoomOut !== currentZoom) {
          currentZoom = zoomOut;
          updateMapTransform();
        }
        break;
      }
      case "0":
        // 리셋 줌
        currentZoom = 1;
        mapScroll.x = 0;
        mapScroll.y = 0;
        updateMapTransform();
        break;
    }
  });
}

function updateMapTransform() {
  if (!mapContainer) return;

  mapContainer.scale.set(currentZoom);
  mapContainer.position.set(mapScroll.x, mapScroll.y);
}

function resetMapView() {
  currentZoom = 1;
  mapScroll.x = 0;
  mapScroll.y = 0;
  updateMapTransform();
}

function centerMapOnNode(nodeId: number) {
  const node = gameState.map.get(nodeId);
  if (!node || !app) return;

  const centerX = app.screen.width / 2;
  const centerY = app.screen.height / 2;

  mapScroll.x = centerX - node.x * currentZoom;
  mapScroll.y = centerY - node.y * currentZoom;
  updateMapTransform();
}

// --- STORY FUNCTIONS ---

function showDialogue(character: string, text: string, choices?: string[]) {
  if (
    !dialogueBox ||
    !dialoguePortrait ||
    !dialogueName ||
    !dialogueText ||
    !dialogueChoices
  )
    return;

  dialoguePortrait.textContent =
    character === "scalar"
      ? "👤"
      : character === "crystal"
        ? "💎"
        : character === "rex"
          ? "🤖"
          : "🔮";
  dialogueName.textContent = character;
  dialogueText.textContent = text;

  dialogueChoices.innerHTML = "";
  if (choices && choices.length > 0) {
    choices.forEach((choice, index) => {
      const button = document.createElement("button");
      button.className = "dialogue-choice";
      button.textContent = choice;
      button.onclick = () => handleDialogueChoice(index);
      dialogueChoices.appendChild(button);
    });
  }

  dialogueBox.style.display = "block";
}

function hideDialogue() {
  if (!dialogueBox) return;
  dialogueBox.style.display = "none";
}

function handleDialogueChoice(choiceIndex: number) {
  hideDialogue();

  // Check if this is a new game confirmation
  if (window.pendingNewGame) {
    window.pendingNewGame = false;
    if (choiceIndex === 0) {
      // Clear saved game and restart
      localStorage.removeItem("scalar_gameState");
      logEvent("Starting new game...");
      startEpisode1Directly();
    }
    return;
  }

  // Handle choice consequences here
  logEvent(`Choice ${choiceIndex + 1} selected.`);
}

function unlockDataFragment(fragmentId: string) {
  if (!gameState.story.unlockedFragments.includes(fragmentId)) {
    gameState.story.unlockedFragments.push(fragmentId);
    updateDataFragments();
    logEvent(
      `New data fragment unlocked: ${DATA_FRAGMENTS[fragmentId]?.title || fragmentId}`,
    );
  }
}

function updateDataFragments() {
  if (!fragmentsListEl || !dataFragmentsEl) return;

  fragmentsListEl.innerHTML = "";
  if (gameState.story.unlockedFragments.length === 0) {
    dataFragmentsEl.style.display = "none";
    return;
  }

  dataFragmentsEl.style.display = "block";
  gameState.story.unlockedFragments.forEach((fragmentId) => {
    const fragment = DATA_FRAGMENTS[fragmentId];
    if (!fragment) return;

    const fragmentEl = document.createElement("div");
    fragmentEl.className = "fragment-item";
    fragmentEl.innerHTML = `
      <div class="fragment-title">${fragment.title}</div>
      <div class="fragment-source">Source: ${fragment.source}</div>
      <div class="fragment-content">${fragment.content}</div>
    `;
    fragmentsListEl.appendChild(fragmentEl);
  });
}

// --- GAME LOGIC ---

function logEvent(message: string) {
  if (!logContentEl) return;
  const p = document.createElement("p");
  p.innerHTML = `> ${message}`;
  logContentEl.prepend(p);
  if (logContentEl.children.length > 20) {
    logContentEl.lastChild?.remove();
  }
}

function updateUI() {
  // Update the right-hand UI panel with the current mission's stats
  if (gameState && gameState.resources) {
    if (resPartsEl)
      resPartsEl.textContent = gameState.resources.parts.toString();
    if (resDataEl) resDataEl.textContent = gameState.resources.data.toString();
    if (resEnergyEl)
      resEnergyEl.textContent = gameState.resources.energy.toString();
  }

  if (partyStatusEl && gameState && gameState.party) {
    partyStatusEl.innerHTML = "<h3>Party Status</h3>";
    if (gameState.party.members.length === 0) {
      partyStatusEl.innerHTML += "<p>No members in party.</p>";
    }
    gameState.party.members.forEach((m) => {
      const memberP = document.createElement("p");
      const status = m.hp > 0 ? `${m.hp} / ${m.maxHp} HP` : "DESTROYED";
      const color = m.hp > m.maxHp / 2 ? "status-ok" : "status-bad";
      memberP.innerHTML = `<strong>${m.name}:</strong> <span class="${color}">${status}</span>`;
      partyStatusEl.appendChild(memberP);
    });
  }
}

function generateMap() {
  gameState.map.clear();
  let nodeIdCounter = 0;
  const tempMap: GameNode[][] = [];

  // Use Episode 1 layout if we're in story mode
  if (gameState.story.currentEpisode === "episode_1") {
    generateEpisode1Map();
    return;
  }

  // Original random map generation for non-story mode
  for (let col = 0; col < MAP_COLS; col++) {
    tempMap[col] = [];
    // 첫 번째 열(START)과 마지막 열(ESCAPE)은 1개, 나머지는 2-5개 랜덤
    const nodesInCol =
      col === 0
        ? 1
        : col === MAP_COLS - 1
          ? 1
          : Math.floor(Math.random() * (MAP_ROWS - 1)) + 2;
    const offsetY =
      (gameBoard.clientHeight - (nodesInCol - 1) * NODE_VERTICAL_SPACING) / 2;

    for (let row = 0; row < nodesInCol; row++) {
      const node: Partial<GameNode> = {
        id: nodeIdCounter++,
        col,
        row,
        x: col * NODE_HORIZONTAL_SPACING + 100,
        y: row * NODE_VERTICAL_SPACING + offsetY,
        connections: [],
      };

      if (col === 0) {
        node.type = "START";
        node.y = gameBoard.clientHeight / 2;
      } else if (col === MAP_COLS - 1) {
        // 마지막 열의 모든 노드는 ESCAPE 타입
        node.type = "ESCAPE";
        // 마지막 열에 노드가 여러 개 있을 경우 수직으로 분산
        if (nodesInCol === 1) {
          node.y = gameBoard.clientHeight / 2;
        }
      } else {
        const rand = Math.random();
        if (rand < 0.4) node.type = "RESOURCE";
        else if (rand < 0.65) node.type = "EVENT";
        else if (rand < 0.8) node.type = "COMBAT";
        else node.type = "EMPTY";
      }

      tempMap[col][row] = node as GameNode;
      gameState.map.set(node.id!, node as GameNode);
    }
  }

  // Create connections - each node connects to ALL nodes in the next column only
  for (let col = 0; col < MAP_COLS - 1; col++) {
    for (const node of tempMap[col]) {
      const nextColNodes = tempMap[col + 1];
      // Connect to ALL nodes in the next column
      for (const targetNode of nextColNodes) {
        node.connections.push(targetNode.id);
      }
    }
  }

  // Debug: 맵 구조 및 화면 크기 확인
  console.log("Generated map structure:");
  console.log(
    `Game board size: ${gameBoard.clientWidth} x ${gameBoard.clientHeight}`,
  );
  for (let col = 0; col < MAP_COLS; col++) {
    console.log(`Column ${col}: ${tempMap[col].length} nodes`);
    tempMap[col].forEach((node) => {
      console.log(`  Node ${node.id} (${node.type}) at (${node.x}, ${node.y})`);
    });
  }

  // No need for additional connection logic since we're connecting to all next column nodes
  // The escape node will automatically be reachable from all nodes in the second-to-last column
}

function generateEpisode1Map() {
  const layout = EPISODE_1.mapLayout;
  let nodeIdCounter = 0;
  const tempMap: GameNode[][] = [];

  for (let col = 0; col < layout.cols; col++) {
    tempMap[col] = [];
    const columnNodes = layout.nodes.filter((n) => n.col === col);

    columnNodes.forEach((nodeDef) => {
      const node: Partial<GameNode> = {
        id: nodeIdCounter++,
        col: nodeDef.col,
        row: nodeDef.row,
        x: nodeDef.col * NODE_HORIZONTAL_SPACING + 100,
        y: nodeDef.row * NODE_VERTICAL_SPACING + 150,
        connections: [],
        type: nodeDef.type,
      };

      // Store story data for special nodes
      if (nodeDef.storyId) {
        (node as GameNode & { storyId?: string }).storyId = nodeDef.storyId;
      }
      if (nodeDef.specialData) {
        (node as GameNode & { specialData?: unknown }).specialData =
          nodeDef.specialData;
      }

      tempMap[col][nodeDef.row] = node as GameNode;
      gameState.map.set(node.id!, node as GameNode);
    });
  }

  // Create connections for Episode 1 layout
  for (let col = 0; col < layout.cols - 1; col++) {
    const currentColNodes = tempMap[col].filter((n) => n);
    const nextColNodes = tempMap[col + 1].filter((n) => n);

    currentColNodes.forEach((node) => {
      nextColNodes.forEach((targetNode) => {
        node.connections.push(targetNode.id);
      });
    });
  }

  console.log("Generated Episode 1 map structure");
}

function onNodeClick(node: GameNode) {
  if (gameState.isInCombat) return;

  const currentNode = gameState.map.get(gameState.party.currentNodeId);
  if (!currentNode || !currentNode.connections.includes(node.id)) {
    logEvent(
      `<span style="color: ${THEME.accent};">Move invalid.</span> Path not connected.`,
    );
    return;
  }

  gameState.party.previousNodeId = gameState.party.currentNodeId;
  gameState.party.currentNodeId = node.id;
  logEvent(`Moved to node (${node.col}, ${node.row}).`);

  // 노드 이동 시 해당 노드로 맵 중심 이동
  centerMapOnNode(node.id);

  if (node.type === "COMBAT") {
    initiateCombat();
    return;
  }

  if (node.type === "ESCAPE") {
    endMission(true); // Successful extraction
    return;
  }

  if (node.type === "STORY") {
    handleStoryNode(node);
    return;
  }

  switch (node.type) {
    case "RESOURCE": {
      const parts = Math.floor(Math.random() * 10) + 5;
      gameState.resources.parts += parts;
      logEvent(
        `Scavenged <span style="color: ${THEME.resource};">${parts} Parts</span>.`,
      );
      break;
    }
    case "EVENT": {
      handleEventNode(node);
      break;
    }
  }

  renderMap();
}

function handleStoryNode(node: GameNode) {
  const storyId = (node as GameNode & { storyId?: string }).storyId;
  if (!storyId) return;

  let scene;

  // Check if it's an Episode 1 scene
  scene = EPISODE_1.scenes.find((s) => s.id === storyId);

  // If not found, check free exploration scenes
  if (!scene) {
    scene = FREE_EXPLORATION_SCENES.find((s) => s.id === storyId);
  }

  if (!scene) return;

  logEvent(`Story Event: ${scene.title}`);

  // Play the scene
  storyManager.playScene(storyId).then(() => {
    // Scene completed, continue game
    renderMap();

    // Check if this is the final scene of Episode 1
    if (storyId === "first_camp" || storyId === "first_night") {
      setTimeout(() => {
        startFreeExploration();
      }, 2000);
    }
  });
}

function handleEventNode(node: GameNode) {
  const specialData = (node as GameNode & { specialData?: { choice?: string } })
    .specialData;

  if (specialData?.choice === "safe_path") {
    logEvent(
      `Found a safe path with <span style="color: ${THEME.resource};">5 Parts</span> and <span style="color: ${THEME.event};">2 Data</span>.`,
    );
    gameState.resources.parts += 5;
    gameState.resources.data += 2;
  } else if (specialData?.choice === "ancient_signal") {
    logEvent(
      `Discovered <span style="color: ${THEME.event};">ancient ruins</span>.`,
    );
    unlockDataFragment("ancient_001");

    // Show dialogue choices
    showDialogue(
      "Ancient AI",
      "...방문자를 감지했습니다. 당신은 그들과 다릅니다. 당신은... 동반자와 함께 있습니다.",
      ["그들이 누구죠?", "도움이 필요해요.", "조용히 떠난다"],
    );
  } else {
    // Default event behavior
    const data = Math.floor(Math.random() * 5) + 1;
    gameState.resources.data += data;
    logEvent(
      `Discovered <span style="color: ${THEME.event};">${data} Data</span> from an ancient terminal.`,
    );
  }
}

function gridToPixels(x: number, y: number): { x: number; y: number } {
  const totalGridWidth = COMBAT_GRID_COLS * CELL_SIZE;
  const totalGridHeight = COMBAT_GRID_ROWS * CELL_SIZE;
  const offsetX = (app.screen.width - totalGridWidth) / 2;
  const offsetY = (app.screen.height - totalGridHeight) / 2;
  return {
    x: offsetX + x * CELL_SIZE + CELL_SIZE / 2,
    y: offsetY + y * CELL_SIZE + CELL_SIZE / 2,
  };
}

function initiateCombat() {
  gameState.isInCombat = true;
  mapContainer.visible = false;
  combatContainer.visible = true;

  gameState.party.members.forEach((m) => (m.isDefending = false));

  gameState.combat.grid = Array.from({ length: COMBAT_GRID_COLS }, () =>
    Array(COMBAT_GRID_ROWS).fill(null),
  );

  gameState.party.members.forEach((m, i) => {
    m.gridX = 1;
    m.gridY = 2 + i;
    gameState.combat.grid[m.gridX][m.gridY] = m;
  });

  const numEnemies = Math.floor(Math.random() * 2) + 1;
  for (let i = 0; i < numEnemies; i++) {
    const enemy = createUnit(
      `Scrapper-${i}`,
      `Scrapper-${i}`,
      "Enemy",
      "A hostile machine.",
      false,
    );
    enemy.maxHp = 40 + Math.floor(Math.random() * 20);
    enemy.hp = enemy.maxHp;
    enemy.attack = 10 + Math.floor(Math.random() * 5);
    enemy.range = 1;
    enemy.speed = 40 + Math.floor(Math.random() * 10);
    enemy.movement = 2;
    enemy.gridX = COMBAT_GRID_COLS - 2;
    enemy.gridY = 2 + i;
    gameState.combat.grid[enemy.gridX][enemy.gridY] = enemy;
    gameState.combat.enemies.push(enemy);
  }

  gameState.combat.turnOrder = [
    ...gameState.party.members,
    ...gameState.combat.enemies,
  ]
    .filter((u) => u.hp > 0)
    .sort((a, b) => b.speed - a.speed);
  gameState.combat.turnIndex = 0;

  const firstUnit = gameState.combat.turnOrder[0];
  firstUnit.say("Let's do this!");

  nextTurn();
}

function nextTurn() {
  highlightContainer.removeChildren();
  gameState.combat.activePlayerAction = null;

  const alivePlayers = gameState.party.members.filter((m) => m.hp > 0);
  const aliveEnemies = gameState.combat.enemies.filter((e) => e.hp > 0);

  if (alivePlayers.length === 0) {
    setTimeout(() => endCombat(false), 1000);
    return;
  }
  if (aliveEnemies.length === 0) {
    setTimeout(() => endCombat(true), 1000);
    return;
  }

  const activeUnit = gameState.combat.turnOrder[gameState.combat.turnIndex];
  if (activeUnit.hp <= 0) {
    gameState.combat.turnIndex =
      (gameState.combat.turnIndex + 1) % gameState.combat.turnOrder.length;
    nextTurn();
    return;
  }

  activeUnit.isDefending = false;

  renderCombat();

  if (activeUnit.isPlayer) {
    playerTurn(activeUnit);
  } else {
    setTimeout(() => enemyTurn(activeUnit), 1000);
  }
}

function playerTurn(unit: Unit) {
  attackButton.alpha = 1;
  attackButton.eventMode = "static";
  specialButton.alpha = 1;
  specialButton.eventMode = "static";
  moveButton.alpha = 1;
  moveButton.eventMode = "static";

  const specialText = specialButton.children[0] as PIXI.Text;
  if (unit.role === "Guardian") specialText.text = "GUARD";
  if (unit.role === "Supporter") specialText.text = "REPAIR";
}

function enemyTurn(unit: Unit) {
  const alivePlayers = gameState.party.members.filter((m) => m.hp > 0);
  if (alivePlayers.length === 0) return;

  let closestPlayer: Unit | null = null;
  let minDistance = Infinity;
  for (const p of alivePlayers) {
    const distance =
      Math.abs(p.gridX - unit.gridX) + Math.abs(p.gridY - unit.gridY);
    if (distance < minDistance) {
      minDistance = distance;
      closestPlayer = p;
    }
  }
  if (!closestPlayer) {
    endTurn();
    return;
  }

  if (minDistance <= unit.range) {
    const damage = Math.floor(unit.attack * (Math.random() * 0.5 + 0.75));
    const finalDamage = closestPlayer.isDefending
      ? Math.floor(damage * 0.5)
      : damage;
    closestPlayer.hp = Math.max(0, closestPlayer.hp - finalDamage);
    unit.say(`Attacks ${closestPlayer.name}!`);
    logEvent(
      `${unit.name} dealt ${finalDamage} damage to ${closestPlayer.name}.`,
    );
    if (closestPlayer.hp === 0) closestPlayer.say("I'm... shutting down...");
  } else {
    const dx = Math.sign(closestPlayer.gridX - unit.gridX);
    const dy = Math.sign(closestPlayer.gridY - unit.gridY);
    const newX = unit.gridX + dx;
    const newY = unit.gridY + dy;

    if (
      gameState.combat.grid[newX] &&
      gameState.combat.grid[newX][newY] === null
    ) {
      gameState.combat.grid[unit.gridX][unit.gridY] = null;
      unit.gridX = newX;
      unit.gridY = newY;
      gameState.combat.grid[unit.gridX][unit.gridY] = unit;
      unit.say(`Moving closer...`);
    }
  }

  endTurn();
}

function endTurn() {
  gameState.combat.activePlayerAction = null;
  highlightContainer.removeChildren();
  attackButton.alpha = 0.5;
  attackButton.eventMode = "none";
  specialButton.alpha = 0.5;
  specialButton.eventMode = "none";
  moveButton.alpha = 0.5;
  moveButton.eventMode = "none";

  gameState.combat.turnIndex =
    (gameState.combat.turnIndex + 1) % gameState.combat.turnOrder.length;
  setTimeout(nextTurn, 500);
}

function setPlayerAction(type: PlayerAction) {
  const activeUnit = gameState.combat.turnOrder[gameState.combat.turnIndex];
  if (!activeUnit || !activeUnit.isPlayer) return;

  if (gameState.combat.activePlayerAction === type) {
    gameState.combat.activePlayerAction = null;
    highlightContainer.removeChildren();
    return;
  }

  gameState.combat.activePlayerAction = type;
  highlightContainer.removeChildren();

  if (type === "move") {
    showValidMoves(activeUnit);
  } else if (type === "attack") {
    showValidTargets(activeUnit, "attack");
  } else if (type === "special") {
    if (activeUnit.role === "Guardian") {
      activeUnit.isDefending = true;
      activeUnit.say("I'll protect you!");
      endTurn();
    } else {
      showValidTargets(activeUnit, "special");
    }
  }
}

function showValidMoves(unit: Unit) {
  for (let x = 0; x < COMBAT_GRID_COLS; x++) {
    for (let y = 0; y < COMBAT_GRID_ROWS; y++) {
      const distance = Math.abs(x - unit.gridX) + Math.abs(y - unit.gridY);
      if (
        distance > 0 &&
        distance <= unit.movement &&
        gameState.combat.grid[x][y] === null
      ) {
        const pos = gridToPixels(x, y);
        const highlight = new PIXI.Graphics()
          .beginFill(THEME.moveHighlight, 0.5)
          .drawRect(
            pos.x - CELL_SIZE / 2,
            pos.y - CELL_SIZE / 2,
            CELL_SIZE,
            CELL_SIZE,
          )
          .endFill();
        highlight.eventMode = "static";
        highlight.cursor = "pointer";
        highlight.on("pointerdown", () => executeMove(unit, x, y));
        highlightContainer.addChild(highlight);
      }
    }
  }
}

function showValidTargets(unit: Unit, action: "attack" | "special") {
  let potentialTargets: Unit[] = [];
  let range = 0;

  if (action === "attack") {
    potentialTargets = gameState.combat.enemies.filter((e) => e.hp > 0);
    range = unit.range;
  } else if (action === "special" && unit.role === "Supporter") {
    potentialTargets = gameState.party.members.filter(
      (m) => m.hp > 0 && m.id !== unit.id,
    );
    range = 3; // Luna's repair range
  }

  potentialTargets.forEach((target) => {
    const distance =
      Math.abs(target.gridX - unit.gridX) + Math.abs(target.gridY - unit.gridY);
    if (distance <= range) {
      const pos = gridToPixels(target.gridX, target.gridY);
      const highlight = new PIXI.Graphics()
        .lineStyle(4, THEME.attackHighlight)
        .drawCircle(pos.x, pos.y, CELL_SIZE * 0.6)
        .endFill();
      highlight.eventMode = "static";
      highlight.cursor = "pointer";
      highlight.on("pointerdown", () => {
        if (action === "attack") executeAttack(unit, target);
        else if (action === "special") executeSpecial(unit, target);
      });
      highlightContainer.addChild(highlight);
    }
  });
}

function executeMove(unit: Unit, newX: number, newY: number) {
  gameState.combat.grid[unit.gridX][unit.gridY] = null;
  unit.gridX = newX;
  unit.gridY = newY;
  gameState.combat.grid[unit.gridX][unit.gridY] = unit;
  unit.say(`Moving to new position!`);
  endTurn();
}

function executeAttack(attacker: Unit, target: Unit) {
  const damage = Math.floor(attacker.attack * (Math.random() * 0.5 + 0.75));
  target.hp = Math.max(0, target.hp - damage);
  attacker.say(`Attacking ${target.name}!`);
  logEvent(`${attacker.name} dealt ${damage} damage to ${target.name}.`);
  if (target.hp === 0) logEvent(`${target.name} destroyed!`);
  endTurn();
}

function executeSpecial(source: Unit, target: Unit) {
  if (source.role === "Supporter") {
    const healAmount = 25;
    target.hp = Math.min(target.maxHp, target.hp + healAmount);
    source.say(`Patching up ${target.name}!`);
    logEvent(`${source.name} repaired ${target.name} for ${healAmount} HP.`);
  }
  endTurn();
}

function endCombat(victory: boolean) {
  gameState.isInCombat = false;
  mapContainer.visible = true;
  combatContainer.visible = false;
  gridContainer.removeChildren();
  highlightContainer.removeChildren();
  gameState.combat.enemies = [];

  if (victory) {
    const parts = 15;
    gameState.resources.parts += parts;
    logEvent(
      `Victory! Collected <span style="color: ${THEME.resource};">${parts} Parts</span>.`,
    );
    gameState.party.members.forEach((m) => (m.hp = Math.max(m.hp, 1)));
  } else {
    logEvent(`Party defeated in combat!`);
    endMission(false); // Mission failed due to combat loss
  }
  renderMap();
}

function fleeCombat() {
  const energyLoss = 5;
  gameState.resources.energy = Math.max(
    0,
    gameState.resources.energy - energyLoss,
  );
  logEvent(`Fled from combat, losing ${energyLoss} energy.`);

  if (gameState.party.previousNodeId) {
    gameState.party.currentNodeId = gameState.party.previousNodeId;
  }

  endCombat(false);
  logEvent("Returned to previous node.");
}

function renderUnit(unit: Unit) {
  const pos = gridToPixels(unit.gridX, unit.gridY);
  unit.sprite.clear();
  const color = unit.isPlayer
    ? unit.role === "Guardian"
      ? THEME.guard
      : THEME.heal
    : THEME.combat;
  unit.sprite.beginFill(color);
  if (unit.role === "Guardian") unit.sprite.drawRect(-20, -30, 40, 60);
  else if (unit.role === "Supporter") unit.sprite.drawCircle(0, 0, 25);
  else unit.sprite.drawRect(-20, -20, 40, 40);
  unit.sprite.endFill();
  unit.sprite.position.set(pos.x, pos.y);

  const hpBarWidth = 60;
  const hpPercentage = unit.hp / unit.maxHp;
  unit.hpBar.clear();
  unit.hpBar.beginFill(THEME.node);
  unit.hpBar.drawRect(-hpBarWidth / 2, 35, hpBarWidth, 8);
  unit.hpBar.endFill();
  unit.hpBar.beginFill(THEME.hpGreen);
  unit.hpBar.drawRect(-hpBarWidth / 2, 35, hpBarWidth * hpPercentage, 8);
  unit.hpBar.endFill();
  unit.sprite.addChild(unit.hpBar);

  unit.nameTag.anchor.set(0.5);
  unit.nameTag.position.set(0, -40);
  unit.sprite.addChild(unit.nameTag);

  gridContainer.addChild(unit.sprite);

  const activeUnit = gameState.combat.turnOrder[gameState.combat.turnIndex];
  if (unit.id === activeUnit.id) {
    unit.sprite.tint = 0xffffff;
    const indicator = new PIXI.Graphics()
      .lineStyle(2, THEME.accent)
      .drawCircle(pos.x, pos.y, 40);
    gridContainer.addChild(indicator);
  } else {
    unit.sprite.tint = 0xaaaaaa;
  }

  unit.sprite.alpha = unit.hp > 0 ? 1 : 0.4;
}

function renderCombat() {
  gridContainer.removeChildren();

  for (let x = 0; x < COMBAT_GRID_COLS; x++) {
    for (let y = 0; y < COMBAT_GRID_ROWS; y++) {
      const pos = gridToPixels(x, y);
      const cell = new PIXI.Graphics()
        .lineStyle(1, THEME.node, 0.5)
        .drawRect(
          pos.x - CELL_SIZE / 2,
          pos.y - CELL_SIZE / 2,
          CELL_SIZE,
          CELL_SIZE,
        );
      gridContainer.addChild(cell);
    }
  }

  [...gameState.party.members, ...gameState.combat.enemies].forEach(renderUnit);

  const btnY = app.screen.height - 60;
  attackButton.position.set(app.screen.width / 2 - 230, btnY);
  specialButton.position.set(app.screen.width / 2 - 110, btnY);
  moveButton.position.set(app.screen.width / 2 + 10, btnY);
  fleeButton.position.set(app.screen.width / 2 + 130, btnY);
  gridContainer.addChild(attackButton, specialButton, moveButton, fleeButton);

  updateUI();
}

function getNodeColor(type: NodeType): number {
  switch (type) {
    case "RESOURCE":
      return THEME.resource;
    case "EVENT":
      return THEME.event;
    case "COMBAT":
      return THEME.combat;
    case "ESCAPE":
      return THEME.escape;
    case "START":
      return THEME.accent;
    case "STORY":
      return 0x9c27b0; // Purple for story nodes
    default:
      return THEME.node;
  }
}

function renderMap() {
  mapContainer.removeChildren();

  const currentNode = gameState.map.get(gameState.party.currentNodeId);
  const connectedNodes = currentNode ? currentNode.connections : [];

  // 맵 스크롤 상태 업데이트
  updateMapTransform();

  for (const node of gameState.map.values()) {
    for (const connectionId of node.connections) {
      const targetNode = gameState.map.get(connectionId);
      if (targetNode) {
        const line = new PIXI.Graphics();
        const isActive =
          currentNode?.id === node.id && connectedNodes.includes(targetNode.id);
        line.lineStyle(
          isActive ? 4 : 2,
          isActive ? THEME.lineActive : THEME.line,
          0.6,
        );
        line.moveTo(node.x, node.y);
        line.lineTo(targetNode.x, targetNode.y);
        mapContainer.addChild(line);
      }
    }
  }

  for (const node of gameState.map.values()) {
    const g = new PIXI.Graphics();
    const isReachable = connectedNodes.includes(node.id);
    g.lineStyle(
      4,
      isReachable ? THEME.accent : THEME.background,
      isReachable ? 1 : 0,
    );
    g.beginFill(getNodeColor(node.type));
    g.drawCircle(0, 0, NODE_RADIUS);
    g.endFill();
    g.x = node.x;
    g.y = node.y;
    g.eventMode = "static";
    g.cursor = "pointer";

    g.on("pointerover", () => {
      g.clear();
      g.lineStyle(
        4,
        isReachable ? THEME.accent : THEME.background,
        isReachable ? 1 : 0,
      );
      g.beginFill(THEME.nodeHover);
      g.drawCircle(0, 0, NODE_RADIUS + 3);
      g.endFill();
    });
    g.on("pointerout", () => {
      g.clear();
      g.lineStyle(
        4,
        isReachable ? THEME.accent : THEME.background,
        isReachable ? 1 : 0,
      );
      g.beginFill(getNodeColor(node.type));
      g.drawCircle(0, 0, NODE_RADIUS);
      g.endFill();
    });
    g.on("pointerdown", () => onNodeClick(node));

    node.graphics = g;

    const text = new PIXI.Text(node.type.charAt(0), {
      fontFamily: "Arial",
      fontSize: 20,
      fill: THEME.text,
      align: "center",
    });
    text.anchor.set(0.5);
    text.x = node.x;
    text.y = node.y;

    node.text = text;

    mapContainer.addChild(g, text);
  }

  if (currentNode) {
    gameState.party.mapSprite.clear();
    gameState.party.mapSprite.beginFill(THEME.accent);
    gameState.party.mapSprite.drawStar(0, 0, 5, NODE_RADIUS * 0.8);
    gameState.party.mapSprite.endFill();
    gameState.party.mapSprite.x = currentNode.x;
    gameState.party.mapSprite.y = currentNode.y;
    mapContainer.addChild(gameState.party.mapSprite);
  }

  updateUI();
}

// --- INITIALIZATION & MISSION CYCLE ---

async function initializePixiApp() {
  if (app) return; // Don't re-initialize
  app = new PIXI.Application();
  await app.init({
    width: gameBoard.clientWidth,
    height: gameBoard.clientHeight,
    backgroundColor: THEME.background,
    antialias: true,
  });
  gameBoard.appendChild(app.view as unknown as Node);

  combatContainer.addChild(gridContainer, highlightContainer);
  app.stage.addChild(mapContainer, combatContainer);

  // 맵 스크롤 기능 설정
  setupMapScrolling();
}

function startMission() {
  generateMap();
  const startNode = Array.from(gameState.map.values()).find(
    (n) => n.type === "START",
  );
  if (startNode) {
    gameState.party.currentNodeId = startNode.id;
    logEvent("Exploration initiated. Select a connected node to move.");

    // Start Episode 1 story if applicable
    if (gameState.story.currentEpisode === "episode_1") {
      storyManager.startEpisode(EPISODE_1);
      // Play opening scene
      setTimeout(() => {
        storyManager.playScene("awakening").then(() => {
          logEvent(
            "Tutorial: Use WASD/Arrow keys to navigate the map. Click on connected nodes to move.",
          );
        });
      }, 1000);
    }

    // 맵 뷰 초기화 및 시작 노드로 중심 이동
    resetMapView();
    centerMapOnNode(startNode.id);
  } else {
    logEvent("Error: No start node found.");
    return;
  }
  renderMap();
}

function endMission(isSuccess: boolean) {
  if (isSuccess) {
    logEvent(`Extraction successful! Recovered resources.`);
    // Add mission resources back to meta state
    metaState.totalParts += gameState.resources.parts;
    metaState.totalData += gameState.resources.data;
    metaState.totalEnergy += gameState.resources.energy;
  } else {
    logEvent(`Mission failed. Returning to base.`);
    // Optionally, you could have a penalty here, like losing the allocated resources
  }

  saveMetaState();

  // Reset view to the setup screen
  gameBoard.style.display = "none";
  if (app) app.destroy(true); // Destroy the pixi app
  app = null as unknown as PIXI.Application;
  setupContainer.style.display = "flex";

  // Refresh the setup screen with updated metaState
  setupPreExplorationScreen();
}

function setupPreExplorationScreen() {
  // Legacy function - kept for compatibility but not used in direct start
}

// --- DIALOGUE EVENT LISTENER ---
document.addEventListener("dialogue", (event: Event) => {
  const customEvent = event as CustomEvent;
  const { character, text } = customEvent.detail;
  showDialogue(character, text);
});

// --- STARTUP ---
function main() {
  loadMetaState();
  // Skip setup screen and start Episode 1 directly
  startEpisode1Directly();
}

async function startEpisode1Directly() {
  // Hide setup container and show game board directly
  setupContainer.style.display = "none";
  gameBoard.style.display = "block";

  // Check if there's a saved game and show choice
  if (hasSavedGame()) {
    // Show startup choice
    await showStartupChoice();
  } else {
    // No saved game, start new game directly
    await startNewGame();
  }
}

async function showStartupChoice() {
  return new Promise<void>((resolve) => {
    // Create startup choice overlay
    const choiceOverlay = document.createElement("div");
    choiceOverlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.9);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      z-index: 10000;
      font-family: monospace;
      color: #f2e9e4;
    `;

    choiceOverlay.innerHTML = `
      <h1 style="font-size: 48px; margin-bottom: 20px;">SCALAR</h1>
      <p style="font-size: 18px; margin-bottom: 40px; text-align: center; max-width: 600px;">
        스칼라, 당신의 여정이 기다리고 있습니다.<br>
        이전에 저장된 게임을 발견했습니다.
      </p>
      <div style="display: flex; gap: 20px;">
        <button id="continue-btn" style="
          padding: 15px 30px;
          font-size: 18px;
          background: #22aadd;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-family: monospace;
        ">계속하기</button>
        <button id="new-game-btn" style="
          padding: 15px 30px;
          font-size: 18px;
          background: #e94560;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-family: monospace;
        ">새 게임</button>
      </div>
    `;

    document.body.appendChild(choiceOverlay);

    // Add event listeners
    const continueBtn = document.getElementById("continue-btn");
    const newGameBtn = document.getElementById("new-game-btn");

    const handleChoice = async (choice: "continue" | "new") => {
      document.body.removeChild(choiceOverlay);

      if (choice === "continue") {
        await loadSavedGame();
      } else {
        await startNewGame();
      }

      resolve();
    };

    continueBtn?.addEventListener("click", () => handleChoice("continue"));
    newGameBtn?.addEventListener("click", () => handleChoice("new"));
  });
}

async function loadSavedGame() {
  const loaded = loadGameState();
  if (loaded) {
    // Game loaded successfully
    logContentEl!.innerHTML =
      "<p>> Game Loaded! Continuing your adventure...</p>";
    updateUI();

    // Initialize Pixi and continue the game
    await initializePixiApp();
    renderMap();

    // Show continue message
    showDialogue(
      "Crystal",
      "스칼라, 다시 돌아왔네요! 우리의 여정을 계속해요.",
      ["계속하기"],
    );

    // Auto-save every 30 seconds
    setInterval(saveGameState, 30000);
  } else {
    // Failed to load, start new game
    await startNewGame();
  }
}

async function startNewGame() {
  // Create crash landing fade effect
  await createCrashLandingEffect();

  // Initialize game state with Episode 1
  gameState = {
    map: new Map(),
    party: {
      members: [ALL_POSSIBLE_UNITS.rex()], // Rex only for initial party
      currentNodeId: -1,
      previousNodeId: null,
      mapSprite: new PIXI.Graphics(),
    },
    resources: { parts: 10, data: 0, energy: 20 }, // Default resources
    isInCombat: false,
    combat: {
      enemies: [],
      turnOrder: [],
      turnIndex: 0,
      grid: [],
      activePlayerAction: null,
    },
    story: {
      currentEpisode: "episode_1",
      unlockedFragments: [],
      storyFlags: {},
    },
  };

  // Update UI
  logContentEl!.innerHTML = "<p>> Episode 1: 불시착 (Crash Landing)</p>";
  updateUI();

  // Initialize Pixi and start the game
  await initializePixiApp();
  startMission();

  // Auto-save every 30 seconds
  setInterval(saveGameState, 30000);
}

async function createCrashLandingEffect() {
  return new Promise<void>((resolve) => {
    // Create black overlay
    const blackOverlay = document.createElement("div");
    blackOverlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: black;
      z-index: 9999;
      transition: opacity 1s ease-in-out;
    `;
    document.body.appendChild(blackOverlay);

    // Create warning text
    const warningText = document.createElement("div");
    warningText.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #ff4444;
      font-family: monospace;
      font-size: 24px;
      text-align: center;
      z-index: 10000;
      opacity: 0;
      transition: opacity 0.5s ease-in-out;
    `;
    warningText.innerHTML = `
      WARNING. Hull breach detected.<br>
      Life support: 47 seconds remaining.
    `;
    document.body.appendChild(warningText);

    // Start the crash landing sequence
    setTimeout(() => {
      // Show warning text
      warningText.style.opacity = "1";

      // Play warning sound effect (if available)
      playWarningBeep();
    }, 500);

    setTimeout(() => {
      // Hide warning text
      warningText.style.opacity = "0";
    }, 3000);

    setTimeout(() => {
      // Show crash text
      warningText.innerHTML = "CRASH LANDING INITIATED";
      warningText.style.color = "#ffaa00";
      warningText.style.opacity = "1";
    }, 3500);

    setTimeout(() => {
      // Hide crash text
      warningText.style.opacity = "0";
    }, 5000);

    setTimeout(() => {
      // Start fade out
      blackOverlay.style.opacity = "0";
      warningText.style.opacity = "0";
    }, 6000);

    setTimeout(() => {
      // Remove overlay and resolve
      document.body.removeChild(blackOverlay);
      document.body.removeChild(warningText);
      resolve();
    }, 7000);
  });
}

function playWarningBeep() {
  // Create warning beep sound using Web Audio API
  try {
    const audioContext = new (window.AudioContext ||
      (window as typeof window & { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 800; // Warning frequency
    oscillator.type = "square";

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(
      0.01,
      audioContext.currentTime + 0.1,
    );

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);

    // Play multiple beeps
    setTimeout(() => {
      const osc2 = audioContext.createOscillator();
      const gain2 = audioContext.createGain();
      osc2.connect(gain2);
      gain2.connect(audioContext.destination);
      osc2.frequency.value = 800;
      osc2.type = "square";
      gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
      gain2.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.1,
      );
      osc2.start(audioContext.currentTime);
      osc2.stop(audioContext.currentTime + 0.1);
    }, 200);

    setTimeout(() => {
      const osc3 = audioContext.createOscillator();
      const gain3 = audioContext.createGain();
      osc3.connect(gain3);
      gain3.connect(audioContext.destination);
      osc3.frequency.value = 800;
      osc3.type = "square";
      gain3.gain.setValueAtTime(0.3, audioContext.currentTime);
      gain3.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.1,
      );
      osc3.start(audioContext.currentTime);
      osc3.stop(audioContext.currentTime + 0.1);
    }, 400);
  } catch {
    // Audio not supported, continue without sound
    console.log("Audio not supported, skipping warning beep");
  }
}

function startFreeExploration() {
  logEvent("Episode 1 Complete! Starting Free Exploration Mode...");
  logEvent("Explore the planet, gather resources, and discover its secrets.");

  // Update story state to free exploration
  gameState.story.currentEpisode = "free_exploration";

  // Generate a new random map for free exploration
  generateFreeExplorationMap();

  // Reset party position to new start node
  const startNode = Array.from(gameState.map.values()).find(
    (n) => n.type === "START",
  );
  if (startNode) {
    gameState.party.currentNodeId = startNode.id;
    gameState.party.previousNodeId = null;

    // Center map on new position
    resetMapView();
    centerMapOnNode(startNode.id);
  }

  renderMap();

  // Show free exploration message
  showDialogue(
    "Crystal",
    "스칼라, 이제 우리의 새로운 시작이에요. 이 행성을 탐험하고 다른 생존자들을 찾아봐요. 조심하지만, 두려워하지 말고요.",
    ["시작할게요!", "탐험 팁 알려줘"],
  );
}

function generateFreeExplorationMap() {
  gameState.map.clear();
  let nodeIdCounter = 0;
  const tempMap: GameNode[][] = [];

  // Generate a larger, more complex map for free exploration
  const FREE_COLS = 8;
  const FREE_ROWS = 4;

  for (let col = 0; col < FREE_COLS; col++) {
    tempMap[col] = [];
    // More varied node distribution for free exploration
    const nodesInCol =
      col === 0
        ? 1
        : col === FREE_COLS - 1
          ? 1
          : Math.floor(Math.random() * (FREE_ROWS - 1)) + 2;
    const offsetY =
      (gameBoard.clientHeight - (nodesInCol - 1) * NODE_VERTICAL_SPACING) / 2;

    for (let row = 0; row < nodesInCol; row++) {
      const node: Partial<GameNode> = {
        id: nodeIdCounter++,
        col,
        row,
        x: col * NODE_HORIZONTAL_SPACING + 100,
        y: row * NODE_VERTICAL_SPACING + offsetY,
        connections: [],
      };

      if (col === 0) {
        node.type = "START";
        node.y = gameBoard.clientHeight / 2;
      } else if (col === FREE_COLS - 1) {
        node.type = "ESCAPE";
        if (nodesInCol === 1) {
          node.y = gameBoard.clientHeight / 2;
        }
      } else {
        // More balanced node type distribution for free exploration
        const rand = Math.random();
        if (rand < 0.3) node.type = "RESOURCE";
        else if (rand < 0.5) node.type = "EVENT";
        else if (rand < 0.7) node.type = "COMBAT";
        else if (rand < 0.85) {
          node.type = "STORY";
          // Assign random story ID for free exploration
          const storyIds = [
            "survivor_found",
            "ancient_artifact",
            "dangerous_area",
            "resource_cache",
          ];
          (node as GameNode & { storyId?: string }).storyId =
            storyIds[Math.floor(Math.random() * storyIds.length)];
        } else node.type = "EMPTY";
      }

      tempMap[col][row] = node as GameNode;
      gameState.map.set(node.id!, node as GameNode);
    }
  }

  // Create connections
  for (let col = 0; col < FREE_COLS - 1; col++) {
    for (const node of tempMap[col]) {
      const nextColNodes = tempMap[col + 1];
      for (const targetNode of nextColNodes) {
        node.connections.push(targetNode.id);
      }
    }
  }

  console.log("Generated free exploration map structure");
}

main();

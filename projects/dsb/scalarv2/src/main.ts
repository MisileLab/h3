import * as PIXI from "pixi.js";

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
type NodeType = "START" | "RESOURCE" | "EVENT" | "COMBAT" | "ESCAPE" | "EMPTY";
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
const launchButton = document.getElementById(
  "launch-button",
) as HTMLButtonElement;
const unitList = document.getElementById("unit-list") as HTMLElement;
const partsInput = document.getElementById("parts-input") as HTMLInputElement;
const energyInput = document.getElementById("energy-input") as HTMLInputElement;
const partsLabel = document.getElementById("parts-label") as HTMLLabelElement;
const energyLabel = document.getElementById("energy-label") as HTMLLabelElement;

const resPartsEl = document.getElementById("res-parts");
const resDataEl = document.getElementById("res-data");
const resEnergyEl = document.getElementById("res-energy");
const logContentEl = document.getElementById("log-content");
const partyStatusEl = document.querySelector("#party-status");

// --- MAP CONTROL ELEMENTS ---
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const zoomResetBtn = document.getElementById("zoom-reset");
const centerMapBtn = document.getElementById("center-map");

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
      const data = Math.floor(Math.random() * 5) + 1;
      gameState.resources.data += data;
      logEvent(
        `Discovered <span style="color: ${THEME.event};">${data} Data</span> from an ancient terminal.`,
      );
      break;
    }
  }

  renderMap();
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
  // Clear previous elements
  unitList.innerHTML = "";

  // Update resource inputs based on metaState
  partsInput.value = "0";
  energyInput.value = "0";
  partsInput.max = metaState.totalParts.toString();
  energyInput.max = metaState.totalEnergy.toString();
  partsLabel.textContent = `Parts: (Available: ${metaState.totalParts})`;
  energyLabel.textContent = `Energy: (Available: ${metaState.totalEnergy})`;

  // Populate unit selection
  metaState.availableUnits.forEach((unit) => {
    const card = document.createElement("div");
    card.className = "unit-card";
    card.dataset.unitId = unit.id;
    card.innerHTML = `
            <h3>${unit.name}</h3>
            <p>${unit.role}</p>
            <p>${unit.description}</p>
        `;
    card.addEventListener("click", () => {
      card.classList.toggle("selected");
    });
    unitList.appendChild(card);
  });
}

async function launchMission() {
  // 1. Gather selections
  const selectedUnitIds = Array.from(
    unitList.querySelectorAll(".unit-card.selected"),
  ).map((card) => (card as HTMLElement).dataset.unitId);

  if (selectedUnitIds.length === 0) {
    alert("Please select at least one unit for the party.");
    return;
  }

  const allocatedParts = parseInt(partsInput.value, 10) || 0;
  const allocatedEnergy = parseInt(energyInput.value, 10) || 0;

  if (
    allocatedParts > metaState.totalParts ||
    allocatedEnergy > metaState.totalEnergy
  ) {
    alert("Cannot allocate more resources than available.");
    return;
  }

  // 2. Create the GameState for this session
  gameState = {
    map: new Map(),
    party: {
      members: metaState.availableUnits
        .filter((u) => selectedUnitIds.includes(u.id))
        .map((u) => ({ ...u })), // Create copies for the mission
      currentNodeId: -1,
      previousNodeId: null,
      mapSprite: new PIXI.Graphics(),
    },
    resources: { parts: allocatedParts, data: 0, energy: allocatedEnergy },
    isInCombat: false,
    combat: {
      enemies: [],
      turnOrder: [],
      turnIndex: 0,
      grid: [],
      activePlayerAction: null,
    },
  };

  // 3. Deduct resources from metaState
  metaState.totalParts -= allocatedParts;
  metaState.totalEnergy -= allocatedEnergy;
  saveMetaState();

  // 4. Update UI panels
  logContentEl!.innerHTML = "<p>> Mission Launched!</p>";
  updateUI();

  // 5. Transition to game view
  setupContainer.style.display = "none";
  gameBoard.style.display = "block";

  // 6. Initialize Pixi and start the game
  await initializePixiApp();
  startMission();
}

// --- STARTUP ---
function main() {
  loadMetaState();
  setupPreExplorationScreen();
  launchButton.onclick = launchMission; // Assign event listener once
}

main();

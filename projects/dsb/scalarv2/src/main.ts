import * as PIXI from 'pixi.js';

// --- THEME & STYLING ---
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
type NodeType = 'START' | 'RESOURCE' | 'EVENT' | 'COMBAT' | 'ESCAPE' | 'EMPTY';
type UnitRole = 'Guardian' | 'Supporter' | 'Scout' | 'Attacker' | 'Enemy';
type PlayerAction = 'move' | 'attack' | 'special';

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
    }
}

// --- UI SELECTORS ---
const resPartsEl = document.getElementById('res-parts');
const resDataEl = document.getElementById('res-data');
const resEnergyEl = document.getElementById('res-energy');
const logContentEl = document.getElementById('log-content');
const partyLocationEl = document.querySelector('#party-status p:nth-child(2)');
const partyStatusEl = document.querySelector('#party-status');


// --- PIXI APP SETUP ---
const gameBoard = document.getElementById('game-board') as HTMLElement;
const app = new PIXI.Application();
await app.init({
    width: gameBoard.clientWidth,
    height: gameBoard.clientHeight,
    backgroundColor: THEME.background,
    antialias: true,
});
gameBoard.appendChild(app.view as unknown as Node);

const mapContainer = new PIXI.Container();
const combatContainer = new PIXI.Container();
const gridContainer = new PIXI.Container();
const highlightContainer = new PIXI.Container();
combatContainer.addChild(gridContainer, highlightContainer);

combatContainer.visible = false;
app.stage.addChild(mapContainer, combatContainer);

// --- GAME STATE FACTORIES ---
const createUnit = (id: string, name: string, role: UnitRole, isPlayer: boolean): Unit => ({
    id,
    name,
    role,
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
        logEvent(`<span style="color: ${this.isPlayer ? THEME.heal : THEME.accent};">${this.name}:</span> "${message}"`);
    }
});

const rex = createUnit('rex', 'Rex', 'Guardian', true);
rex.maxHp = 150; rex.hp = 150; rex.attack = 12; rex.range = 1; rex.speed = 30; rex.movement = 2;

const luna = createUnit('luna', 'Luna', 'Supporter', true);
luna.maxHp = 80; luna.hp = 80; luna.attack = 8; luna.range = 3; luna.speed = 50; luna.movement = 3;

// --- GAME STATE ---
const gameState: GameState = {
    map: new Map(),
    party: {
        members: [rex, luna],
        currentNodeId: -1,
        previousNodeId: null,
        mapSprite: new PIXI.Graphics(),
    },
    resources: { parts: 10, data: 5, energy: 20 },
    isInCombat: false,
    combat: {
        enemies: [],
        turnOrder: [],
        turnIndex: 0,
        grid: [],
        activePlayerAction: null,
    }
};

// --- COMBAT UI ELEMENTS ---
const attackButton = new PIXI.Graphics().beginFill(THEME.accent).drawRoundedRect(0, 0, 100, 50, 10).endFill();
const specialButton = new PIXI.Graphics().beginFill(THEME.resource).drawRoundedRect(0, 0, 100, 50, 10).endFill();
const moveButton = new PIXI.Graphics().beginFill(THEME.guard).drawRoundedRect(0, 0, 100, 50, 10).endFill();
const fleeButton = new PIXI.Graphics().beginFill(THEME.node).drawRoundedRect(0, 0, 100, 50, 10).endFill();

const setupButton = (btn: PIXI.Graphics, text: string, action: () => void) => {
    const btnText = new PIXI.Text(text, { fill: THEME.text, fontSize: 18, align: 'center' });
    btn.addChild(btnText);
    btnText.anchor.set(0.5);
    btnText.position.set(50, 25);
    btn.eventMode = 'static';
    btn.cursor = 'pointer';
    btn.on('pointerdown', action);
    return btn;
};

setupButton(attackButton, 'ATTACK', () => setPlayerAction('attack'));
setupButton(specialButton, 'SPECIAL', () => setPlayerAction('special'));
setupButton(moveButton, 'MOVE', () => setPlayerAction('move'));
setupButton(fleeButton, 'FLEE', () => fleeCombat());


// --- GAME LOGIC ---

function logEvent(message: string) {
    if (!logContentEl) return;
    const p = document.createElement('p');
    p.innerHTML = `> ${message}`;
    logContentEl.prepend(p);
    if (logContentEl.children.length > 20) {
        logContentEl.lastChild?.remove();
    }
}

function updateUI() {
    if (resPartsEl) resPartsEl.textContent = gameState.resources.parts.toString();
    if (resDataEl) resDataEl.textContent = gameState.resources.data.toString();
    if (resEnergyEl) resEnergyEl.textContent = gameState.resources.energy.toString();

    if (partyStatusEl) {
        partyStatusEl.innerHTML = '<h3>Party Status</h3>';
        gameState.party.members.forEach(m => {
            const memberP = document.createElement('p');
            const status = m.hp > 0 ? `${m.hp} / ${m.maxHp} HP` : 'DESTROYED';
            const color = m.hp > m.maxHp / 2 ? 'status-ok' : 'status-bad';
            memberP.innerHTML = `<strong>${m.name}:</strong> <span class="${color}">${status}</span>`;
            partyStatusEl.appendChild(memberP);
        });
    }

    const currentNode = gameState.map.get(gameState.party.currentNodeId);
    if (partyLocationEl && currentNode) {
        partyLocationEl.textContent = `Location: (${currentNode.col}, ${currentNode.row})`;
    }
}

function generateMap() {
    // ... (same as before, no changes needed here)
    let nodeIdCounter = 0;
    const tempMap: GameNode[][] = [];

    for (let col = 0; col < MAP_COLS; col++) {
        tempMap[col] = [];
        const nodesInCol = col === 0 || col === MAP_COLS - 1 ? 1 : Math.floor(Math.random() * (MAP_ROWS -1)) + 2;
        const offsetY = (gameBoard.clientHeight - (nodesInCol - 1) * NODE_VERTICAL_SPACING) / 2;

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
                node.type = 'START';
                node.y = gameBoard.clientHeight / 2;
            } else if (col === MAP_COLS - 1) {
                node.type = 'ESCAPE';
                 node.y = gameBoard.clientHeight / 2;
            } else {
                const rand = Math.random();
                if (rand < 0.4) node.type = 'RESOURCE';
                else if (rand < 0.65) node.type = 'EVENT';
                else if (rand < 0.8) node.type = 'COMBAT';
                else node.type = 'EMPTY';
            }
            
            tempMap[col][row] = node as GameNode;
            gameState.map.set(node.id!, node as GameNode);
        }
    }

    // Create connections
    for (let col = 0; col < MAP_COLS - 1; col++) {
        for (const node of tempMap[col]) {
            const nextColNodes = tempMap[col + 1];
            if (nextColNodes.length > 0) {
                const connections = Math.random() > 0.5 ? 1 : 2;
                for(let i = 0; i < connections; i++) {
                    const targetNode = nextColNodes[Math.floor(Math.random() * nextColNodes.length)];
                    if(!node.connections.includes(targetNode.id)) {
                       node.connections.push(targetNode.id);
                    }
                }
            }
        }
    }
     // Ensure at least one path
    for (let col = 0; col < MAP_COLS - 1; col++) {
        const node = tempMap[col][Math.floor(Math.random() * tempMap[col].length)];
        if (node.connections.length === 0) {
            const nextColNodes = tempMap[col + 1];
            const targetNode = nextColNodes[Math.floor(Math.random() * nextColNodes.length)];
            node.connections.push(targetNode.id);
        }
    }
}


function onNodeClick(node: GameNode) {
    if (gameState.isInCombat) return;

    const currentNode = gameState.map.get(gameState.party.currentNodeId);
    if (!currentNode || !currentNode.connections.includes(node.id)) {
        logEvent(`<span style="color: ${THEME.accent};">Move invalid.</span> Path not connected.`);
        return;
    }
    
    gameState.party.previousNodeId = gameState.party.currentNodeId;
    gameState.party.currentNodeId = node.id;
    logEvent(`Moved to node (${node.col}, ${node.row}).`);

    if (node.type === 'COMBAT') {
        initiateCombat();
        return;
    }
    
    // Node event
    switch(node.type) {
        case 'RESOURCE':
            const parts = Math.floor(Math.random() * 10) + 5;
            gameState.resources.parts += parts;
            logEvent(`Scavenged <span style="color: ${THEME.resource};">${parts} Parts</span>.`);
            break;
        case 'EVENT':
            const data = Math.floor(Math.random() * 5) + 1;
            gameState.resources.data += data;
            logEvent(`Discovered <span style="color: ${THEME.event};">${data} Data</span> from an ancient terminal.`);
            break;
        case 'ESCAPE':
            logEvent(`Extraction successful! Mission complete.`);
            break;
    }

    renderMap();
}

// --- COMBAT LOGIC ---

function gridToPixels(x: number, y: number): { x: number, y: number } {
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
    
    gameState.party.members.forEach(m => m.isDefending = false);

    // Create grid
    gameState.combat.grid = Array.from({ length: COMBAT_GRID_COLS }, () => Array(COMBAT_GRID_ROWS).fill(null));

    // Place units
    gameState.party.members.forEach((m, i) => {
        m.gridX = 1;
        m.gridY = 2 + i;
        gameState.combat.grid[m.gridX][m.gridY] = m;
    });

    const numEnemies = Math.floor(Math.random() * 2) + 1;
    for (let i = 0; i < numEnemies; i++) {
        const enemy = createUnit(`Scrapper-${i}`, `Scrapper-${i}`, 'Enemy', false);
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
    
    gameState.combat.turnOrder = [...gameState.party.members, ...gameState.combat.enemies]
        .filter(u => u.hp > 0)
        .sort((a, b) => b.speed - a.speed);
    gameState.combat.turnIndex = 0;

    const firstUnit = gameState.combat.turnOrder[0];
    firstUnit.say("Let's do this!");

    nextTurn();
}

function nextTurn() {
    highlightContainer.removeChildren();
    gameState.combat.activePlayerAction = null;

    const alivePlayers = gameState.party.members.filter(m => m.hp > 0);
    const aliveEnemies = gameState.combat.enemies.filter(e => e.hp > 0);

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
        gameState.combat.turnIndex = (gameState.combat.turnIndex + 1) % gameState.combat.turnOrder.length;
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
    attackButton.alpha = 1; attackButton.eventMode = 'static';
    specialButton.alpha = 1; specialButton.eventMode = 'static';
    moveButton.alpha = 1; moveButton.eventMode = 'static';
    
    const specialText = specialButton.children[0] as PIXI.Text;
    if (unit.role === 'Guardian') specialText.text = 'GUARD';
    if (unit.role === 'Supporter') specialText.text = 'REPAIR';
}

function enemyTurn(unit: Unit) {
    const alivePlayers = gameState.party.members.filter(m => m.hp > 0);
    if (alivePlayers.length === 0) return;

    let closestPlayer: Unit | null = null;
    let minDistance = Infinity;
    for (const p of alivePlayers) {
        const distance = Math.abs(p.gridX - unit.gridX) + Math.abs(p.gridY - unit.gridY);
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
        const finalDamage = closestPlayer.isDefending ? Math.floor(damage * 0.5) : damage;
        closestPlayer.hp = Math.max(0, closestPlayer.hp - finalDamage);
        unit.say(`Attacks ${closestPlayer.name}!`);
        logEvent(`${unit.name} dealt ${finalDamage} damage to ${closestPlayer.name}.`);
        if (closestPlayer.hp === 0) closestPlayer.say("I'm... shutting down...");
    } else {
        const dx = Math.sign(closestPlayer.gridX - unit.gridX);
        const dy = Math.sign(closestPlayer.gridY - unit.gridY);
        const newX = unit.gridX + dx;
        const newY = unit.gridY + dy;

        if (gameState.combat.grid[newX] && gameState.combat.grid[newX][newY] === null) {
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
    attackButton.alpha = 0.5; attackButton.eventMode = 'none';
    specialButton.alpha = 0.5; specialButton.eventMode = 'none';
    moveButton.alpha = 0.5; moveButton.eventMode = 'none';

    gameState.combat.turnIndex = (gameState.combat.turnIndex + 1) % gameState.combat.turnOrder.length;
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

    if (type === 'move') {
        showValidMoves(activeUnit);
    } else if (type === 'attack') {
        showValidTargets(activeUnit, 'attack');
    } else if (type === 'special') {
        if (activeUnit.role === 'Guardian') {
            activeUnit.isDefending = true;
            activeUnit.say("I'll protect you!");
            endTurn();
        } else {
            showValidTargets(activeUnit, 'special');
        }
    }
}

function showValidMoves(unit: Unit) {
    for (let x = 0; x < COMBAT_GRID_COLS; x++) {
        for (let y = 0; y < COMBAT_GRID_ROWS; y++) {
            const distance = Math.abs(x - unit.gridX) + Math.abs(y - unit.gridY);
            if (distance > 0 && distance <= unit.movement && gameState.combat.grid[x][y] === null) {
                const pos = gridToPixels(x, y);
                const highlight = new PIXI.Graphics()
                    .beginFill(THEME.moveHighlight, 0.5)
                    .drawRect(pos.x - CELL_SIZE / 2, pos.y - CELL_SIZE / 2, CELL_SIZE, CELL_SIZE)
                    .endFill();
                highlight.eventMode = 'static';
                highlight.cursor = 'pointer';
                highlight.on('pointerdown', () => executeMove(unit, x, y));
                highlightContainer.addChild(highlight);
            }
        }
    }
}

function showValidTargets(unit: Unit, action: 'attack' | 'special') {
    let potentialTargets: Unit[] = [];
    let range = 0;

    if (action === 'attack') {
        potentialTargets = gameState.combat.enemies.filter(e => e.hp > 0);
        range = unit.range;
    } else if (action === 'special' && unit.role === 'Supporter') {
        potentialTargets = gameState.party.members.filter(m => m.hp > 0 && m.id !== unit.id);
        range = 3; // Luna's repair range
    }

    potentialTargets.forEach(target => {
        const distance = Math.abs(target.gridX - unit.gridX) + Math.abs(target.gridY - unit.gridY);
        if (distance <= range) {
            const pos = gridToPixels(target.gridX, target.gridY);
            const highlight = new PIXI.Graphics()
                .lineStyle(4, THEME.attackHighlight)
                .drawCircle(pos.x, pos.y, CELL_SIZE * 0.6)
                .endFill();
            highlight.eventMode = 'static';
            highlight.cursor = 'pointer';
            highlight.on('pointerdown', () => {
                if (action === 'attack') executeAttack(unit, target);
                else if (action === 'special') executeSpecial(unit, target);
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
    if (source.role === 'Supporter') {
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
        logEvent(`Victory! Collected <span style="color: ${THEME.resource};">${parts} Parts</span>.`);
        gameState.party.members.forEach(m => m.hp = Math.max(m.hp, 1));
    } else {
        const energyLoss = 20;
        gameState.resources.energy = Math.max(0, gameState.resources.energy - energyLoss);
        const startNode = Array.from(gameState.map.values()).find(n => n.type === 'START')!;
        gameState.party.currentNodeId = startNode.id;
        logEvent(`Party defeated! Lost ${energyLoss} energy. Returning to start.`);
        gameState.party.members.forEach(m => m.hp = m.maxHp / 2);
    }
    renderMap();
}

function fleeCombat() {
    const energyLoss = 5;
    gameState.resources.energy = Math.max(0, gameState.resources.energy - energyLoss);
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
    const color = unit.isPlayer ? (unit.role === 'Guardian' ? THEME.guard : THEME.heal) : THEME.combat;
    unit.sprite.beginFill(color);
    if (unit.role === 'Guardian') unit.sprite.drawRect(-20, -30, 40, 60);
    else if (unit.role === 'Supporter') unit.sprite.drawCircle(0, 0, 25);
    else unit.sprite.drawRect(-20, -20, 40, 40);
    unit.sprite.endFill();
    unit.sprite.position.set(pos.x, pos.y);
    
    const hpBarWidth = 60;
    const hpPercentage = unit.hp / unit.maxHp;
    unit.hpBar.clear();
    unit.hpBar.beginFill(THEME.node);
    unit.hpBar.drawRect(-hpBarWidth/2, 35, hpBarWidth, 8);
    unit.hpBar.endFill();
    unit.hpBar.beginFill(THEME.hpGreen);
    unit.hpBar.drawRect(-hpBarWidth/2, 35, hpBarWidth * hpPercentage, 8);
    unit.hpBar.endFill();
    unit.sprite.addChild(unit.hpBar);

    unit.nameTag.anchor.set(0.5);
    unit.nameTag.position.set(0, -40);
    unit.sprite.addChild(unit.nameTag);

    gridContainer.addChild(unit.sprite);
    
    const activeUnit = gameState.combat.turnOrder[gameState.combat.turnIndex];
    if (unit.id === activeUnit.id) {
        unit.sprite.tint = 0xFFFFFF;
        const indicator = new PIXI.Graphics().lineStyle(2, THEME.accent).drawCircle(pos.x, pos.y, 40);
        gridContainer.addChild(indicator);
    } else {
        unit.sprite.tint = 0xAAAAAA;
    }
    
    unit.sprite.alpha = unit.hp > 0 ? 1 : 0.4;
}

function renderCombat() {
    gridContainer.removeChildren();

    // Draw Grid
    for (let x = 0; x < COMBAT_GRID_COLS; x++) {
        for (let y = 0; y < COMBAT_GRID_ROWS; y++) {
            const pos = gridToPixels(x, y);
            const cell = new PIXI.Graphics()
                .lineStyle(1, THEME.node, 0.5)
                .drawRect(pos.x - CELL_SIZE / 2, pos.y - CELL_SIZE / 2, CELL_SIZE, CELL_SIZE);
            gridContainer.addChild(cell);
        }
    }

    // Render units
    [...gameState.party.members, ...gameState.combat.enemies].forEach(renderUnit);
    
    // Render buttons
    const btnY = app.screen.height - 60;
    attackButton.position.set(app.screen.width / 2 - 230, btnY);
    specialButton.position.set(app.screen.width / 2 - 110, btnY);
    moveButton.position.set(app.screen.width / 2 + 10, btnY);
    fleeButton.position.set(app.screen.width / 2 + 130, btnY);
    gridContainer.addChild(attackButton, specialButton, moveButton, fleeButton);
    
    updateUI();
}


// --- MAP RENDERING ---

function getNodeColor(type: NodeType): number {
    switch (type) {
        case 'RESOURCE': return THEME.resource;
        case 'EVENT': return THEME.event;
        case 'COMBAT': return THEME.combat;
        case 'ESCAPE': return THEME.escape;
        case 'START': return THEME.accent;
        default: return THEME.node;
    }
}

function renderMap() {
    mapContainer.removeChildren();

    const currentNode = gameState.map.get(gameState.party.currentNodeId);
    const connectedNodes = currentNode ? currentNode.connections : [];

    // Draw lines
    for (const node of gameState.map.values()) {
        for (const connectionId of node.connections) {
            const targetNode = gameState.map.get(connectionId);
            if (targetNode) {
                const line = new PIXI.Graphics();
                const isActive = currentNode?.id === node.id && connectedNodes.includes(targetNode.id);
                line.lineStyle(isActive ? 4 : 2, isActive ? THEME.lineActive : THEME.line, 0.6);
                line.moveTo(node.x, node.y);
                line.lineTo(targetNode.x, targetNode.y);
                mapContainer.addChild(line);
            }
        }
    }

    // Draw nodes
    for (const node of gameState.map.values()) {
        const g = new PIXI.Graphics();
        const isReachable = connectedNodes.includes(node.id);
        g.lineStyle(4, isReachable ? THEME.accent : THEME.background, isReachable ? 1 : 0);
        g.beginFill(getNodeColor(node.type));
        g.drawCircle(0, 0, NODE_RADIUS);
        g.endFill();
        g.x = node.x;
        g.y = node.y;
        g.eventMode = 'static';
        g.cursor = 'pointer';
        
        g.on('pointerover', () => {
            g.clear();
            g.lineStyle(4, isReachable ? THEME.accent : THEME.background, isReachable ? 1 : 0);
            g.beginFill(THEME.nodeHover);
            g.drawCircle(0, 0, NODE_RADIUS + 3);
            g.endFill();
        });
        g.on('pointerout', () => {
            g.clear();
            g.lineStyle(4, isReachable ? THEME.accent : THEME.background, isReachable ? 1 : 0);
            g.beginFill(getNodeColor(node.type));
            g.drawCircle(0, 0, NODE_RADIUS);
            g.endFill();
        });
        g.on('pointerdown', () => onNodeClick(node));

        node.graphics = g;

        const text = new PIXI.Text(node.type.charAt(0), {
            fontFamily: 'Arial',
            fontSize: 20,
            fill: THEME.text,
            align: 'center',
        });
        text.anchor.set(0.5);
        text.x = node.x;
        text.y = node.y;
        
        node.text = text;
        
        mapContainer.addChild(g, text);
    }
    
    // Draw party on map
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

// --- INITIALIZATION ---
function initGame() {
    generateMap();
    const startNode = Array.from(gameState.map.values()).find(n => n.type === 'START');
    if (startNode) {
        gameState.party.currentNodeId = startNode.id;
        logEvent("Exploration initiated. Select a connected node to move.");
    } else {
        logEvent("Error: No start node found.");
        return;
    }
    renderMap();
}

initGame();
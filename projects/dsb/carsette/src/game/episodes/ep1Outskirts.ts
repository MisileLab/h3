import { BlueprintConfig, EpisodeConfig } from './types';

const blueprint: BlueprintConfig = {
  id: 'EP1_BLUEPRINT_YARD',
  scrapCostPerMachine: 2,
  productionGoal: {
    basicAttackCassette: 1,
    patchKit: 1,
  },
};

export const EP1_OUTSKIRTS: EpisodeConfig = {
  id: 'EP1_OUTSKIRTS',
  name: 'EP1: OUTSKIRTS',
  startNodeId: 'N0',
  nodes: {
    N0: {
      id: 'N0',
      name: 'GATEHOUSE SKIRMISH',
      type: 'COMBAT',
      next: ['N1'],
      encounter: {
        id: 'EP1_N0_GATEHOUSE',
        mode: 'GATEHOUSE',
        width: 10,
        height: 8,
        obstacles: [{ x: 4, y: 4 }],
        alliesSpawn: [{ x: 2, y: 2 }, { x: 2, y: 3 }],
        enemies: [{ id: 'E0', type: 'BLOCKER', pos: { x: 7, y: 5 } }],
        interactables: [{ id: 'C0', type: 'CHEST_SCRAP', pos: { x: 5, y: 2 }, scrapAmount: 2 }],
      },
    },
    N1: {
      id: 'N1',
      name: 'SUBSTATION',
      type: 'COMBAT',
      next: ['N2'],
      encounter: {
        id: 'EP1_N1_SUBSTATION',
        mode: 'SUBSTATION',
        width: 10,
        height: 8,
        obstacles: [{ x: 5, y: 3 }, { x: 5, y: 4 }],
        alliesSpawn: [{ x: 2, y: 5 }, { x: 2, y: 6 }],
        enemies: [
          { id: 'E0', type: 'BLOCKER', pos: { x: 7, y: 5 } },
          { id: 'E1', type: 'JAMMER', pos: { x: 7, y: 2 } },
        ],
        interactables: [
          { id: 'C0', type: 'CHEST_SCRAP', pos: { x: 4, y: 1 }, scrapAmount: 2 },
          { id: 'B0', type: 'BREAKER', pos: { x: 5, y: 6 } },
        ],
      },
    },
    N2: {
      id: 'N2',
      name: 'MAINTENANCE YARD',
      type: 'BLUEPRINT',
      next: ['N3A', 'N3B'],
      blueprint,
    },
    N3A: {
      id: 'N3A',
      name: 'EXTRACT A // GATEHOUSE UPLINK',
      type: 'COMBAT',
      next: ['RESULT'],
      encounter: {
        id: 'EP1_N3A_EXTRACT_A',
        mode: 'EXTRACT_A',
        width: 10,
        height: 8,
        obstacles: [{ x: 4, y: 2 }, { x: 4, y: 3 }],
        alliesSpawn: [{ x: 2, y: 4 }, { x: 2, y: 5 }],
        enemies: [{ id: 'E0', type: 'BLOCKER', pos: { x: 7, y: 4 }, weakened: true }],
        interactables: [{ id: 'X0', type: 'EXTRACT_CONSOLE', pos: { x: 5, y: 1 } }],
      },
    },
    N3B: {
      id: 'N3B',
      name: 'EXTRACT B // RELAY TOWER UPLINK',
      type: 'COMBAT',
      next: ['RESULT'],
      encounter: {
        id: 'EP1_N3B_EXTRACT_B',
        mode: 'EXTRACT_B',
        width: 10,
        height: 8,
        obstacles: [{ x: 5, y: 1 }, { x: 5, y: 2 }, { x: 5, y: 3 }],
        alliesSpawn: [{ x: 2, y: 6 }, { x: 2, y: 7 }],
        enemies: [
          { id: 'E0', type: 'BLOCKER', pos: { x: 7, y: 6 } },
          { id: 'E1', type: 'BLOCKER', pos: { x: 7, y: 3 } },
          { id: 'E2', type: 'JAMMER', pos: { x: 6, y: 1 } },
        ],
        interactables: [{ id: 'X0', type: 'EXTRACT_CONSOLE', pos: { x: 4, y: 1 } }],
      },
    },
    RESULT: {
      id: 'RESULT',
      name: 'RUN RESULT',
      type: 'RESULT',
      next: [],
    },
  },

};

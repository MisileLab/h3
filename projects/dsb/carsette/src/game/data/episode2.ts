import { EpisodeConfig } from '../types/Run';

export const episode2: EpisodeConfig = {
  id: 'ep2',
  name: 'THE SIGNAL — WORKSHOP LOCKDOWN',
  nodes: [
    {
      id: 'N0',
      name: 'LOADING BAY',
      description: 'Hunter ambush at the loading bay. Learn POUNCE telegraph.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'blocker-n0', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'hunter-n0', name: 'HUNTER', kind: 'HUNTER', maxHp: 7, move: 5, range: 1, damage: 2, ap: 2 },
        ],
      },
    },
    {
      id: 'N1',
      name: 'MAINTENANCE CAGE',
      description: 'Free the trapped student and survive Jammer interference.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'jammer-n1', name: 'JAMMER', kind: 'JAMMER', maxHp: 4, move: 4, range: 4, damage: 1, ap: 2 },
          { id: 'blocker-n1', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
        ],
      },
    },
    {
      id: 'N2',
      name: 'MAKER FLOOR',
      description: 'Safe workshop. Harvest Resonant Dew and craft Coolant.',
      type: 'blueprint',
    },
    {
      id: 'N3',
      name: 'RELAY ROOM',
      description: 'Lockdown core with boosted Hunter. Secure the Relay Core.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'hunter-n3', name: 'HUNTER', kind: 'HUNTER', maxHp: 9, move: 5, range: 1, damage: 2, ap: 2 },
          { id: 'jammer-n3', name: 'JAMMER', kind: 'JAMMER', maxHp: 4, move: 4, range: 4, damage: 1, ap: 2 },
          { id: 'blocker-n3', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
        ],
      },
    },
    {
      id: 'N4A',
      name: 'EXTRACT A // LOADING BAY EXIT',
      description: 'Short safer extraction route.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'blocker-n4a', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 4, move: 3, range: 5, damage: 1, ap: 2 },
        ],
        extraction: {
          console: { x: 6, y: 1 },
          zoneRadius: 1,
          baseStages: 2,
          branch: 'A',
        },
      },
    },
    {
      id: 'N4B',
      name: 'EXTRACT B // ROOFTOP ANTENNA',
      description: 'Long route with Hunter and reinforcements under high HEAT.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'blocker-n4b-1', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'blocker-n4b-2', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'hunter-n4b', name: 'HUNTER', kind: 'HUNTER', maxHp: 7, move: 5, range: 1, damage: 2, ap: 2 },
        ],
        extraction: {
          console: { x: 7, y: 7 },
          zoneRadius: 1,
          baseStages: 3,
          branch: 'B',
        },
      },
    },
    {
      id: 'RESULT',
      name: 'RESULT',
      description: 'Run debrief.',
      type: 'result',
    },
  ],
};

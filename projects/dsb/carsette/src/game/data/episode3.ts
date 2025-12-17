import { EpisodeConfig } from '../types/Run';

export const episode3: EpisodeConfig = {
  id: 'ep3',
  name: 'THE FIRST RULE — SHELTER BREACH',
  nodes: [
    {
      id: 'N0',
      name: 'SHELTER HUB',
      description: 'Rule declared. Alarm at the shelter gym. Prep for breach.',
      type: 'dialogue',
    },
    {
      id: 'N1',
      name: 'CLINIC RUN',
      description: 'Breach team pushes toward the clinic to reach the trapped medic.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'blocker-n1', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'jammer-n1', name: 'JAMMER', kind: 'JAMMER', maxHp: 4, move: 4, range: 4, damage: 1, ap: 2 },
        ],
        resonanceNodes: [
          { x: 5, y: 2 },
          { x: 6, y: 5 },
        ],
      },
    },
    {
      id: 'N2',
      name: 'RESCUE & TRUST',
      description: 'Free the medic, declare the first rule, choose how to trust.',
      type: 'blueprint',
    },
    {
      id: 'N3',
      name: 'ANCHOR APPROACH',
      description: 'Hunter screens the path to the anchor room.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'hunter-n3', name: 'HUNTER', kind: 'HUNTER', maxHp: 8, move: 5, range: 1, damage: 2, ap: 2 },
          { id: 'blocker-n3', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
        ],
        resonanceNodes: [
          { x: 4, y: 1 },
          { x: 6, y: 6 },
          { x: 2, y: 5 },
        ],
      },
    },
    {
      id: 'N4',
      name: 'ANCHOR ROOM',
      description: 'Anchor Warden defends the invasion node. Disable the anchor.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'warden-n4', name: 'ANCHOR WARDEN', kind: 'WARDEN', maxHp: 10, move: 3, range: 5, damage: 2, ap: 2 },
          { id: 'jammer-n4', name: 'JAMMER', kind: 'JAMMER', maxHp: 4, move: 4, range: 4, damage: 1, ap: 2 },
        ],
        resonanceNodes: [
          { x: 3, y: 2 },
          { x: 5, y: 5 },
          { x: 6, y: 2 },
          { x: 2, y: 6 },
        ],
        anchorConsole: { x: 4, y: 4 },
        anchorStagesRequired: 3,
      },
    },
    {
      id: 'N5A',
      name: 'EXTRACT A // UNDERPASS',
      description: 'Safer underpass extraction.',
      type: 'combat',
      encounter: {
        enemies: [
          { id: 'blocker-n5a', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 5, move: 3, range: 5, damage: 1, ap: 2 },
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
      id: 'N5B',
      name: 'EXTRACT B // ROOFTOP',
      description: 'Risky rooftop escape. Surge risk stays high.',
      type: 'combat',
      extractionHeatSpike: 2,
      encounter: {
        enemies: [
          { id: 'blocker-n5b-1', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'blocker-n5b-2', name: 'BLOCKER', kind: 'BLOCKER', maxHp: 6, move: 3, range: 5, damage: 1, ap: 2 },
          { id: 'hunter-n5b', name: 'HUNTER', kind: 'HUNTER', maxHp: 8, move: 5, range: 1, damage: 2, ap: 2 },
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
      description: 'Escort success debrief.',
      type: 'result',
    },
  ],
};

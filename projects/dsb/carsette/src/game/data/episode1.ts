import { EpisodeConfig } from '../types/Run';

export const episode1: EpisodeConfig = {
  id: 'ep1',
  name: 'OUTSKIRTS',
  nodes: [
    {
      id: 'N0',
      name: 'GATEHOUSE SKIRMISH',
      description: 'Tutorial skirmish with a lone Blocker and scrap cache.',
      type: 'combat',
      encounter: {
        enemies: [
          {
            id: 'blocker-0',
            name: 'BLOCKER',
            kind: 'BLOCKER',
            maxHp: 6,
            move: 3,
            range: 5,
            damage: 1,
            ap: 2,
          },
        ],
      },
    },
    {
      id: 'N1',
      name: 'SUBSTATION',
      description: 'Restore power by defending breaker uploads against Blocker and Jammer.',
      type: 'combat',
      encounter: {
        enemies: [
          {
            id: 'blocker-1',
            name: 'BLOCKER',
            kind: 'BLOCKER',
            maxHp: 6,
            move: 3,
            range: 5,
            damage: 1,
            ap: 2,
          },
          {
            id: 'jammer-1',
            name: 'JAMMER',
            kind: 'JAMMER',
            maxHp: 4,
            move: 4,
            range: 4,
            damage: 1,
            ap: 2,
          },
        ],
        extraction: {
          console: { x: 6, y: 6 },
          zoneRadius: 1,
          baseStages: 2,
          branch: 'A',
        },
      },
    },
    {
      id: 'N2',
      name: 'MAINTENANCE YARD',
      description: 'Safe blueprint hub to build Extractor → Refiner → Press and craft items.',
      type: 'blueprint',
    },
    {
      id: 'N3A',
      name: 'EXTRACT A',
      description: 'Gatehouse uplink, lower risk extraction.',
      type: 'combat',
      encounter: {
        enemies: [
          {
            id: 'blocker-2',
            name: 'BLOCKER',
            kind: 'BLOCKER',
            maxHp: 4,
            move: 3,
            range: 5,
            damage: 1,
            ap: 2,
          },
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
      id: 'N3B',
      name: 'EXTRACT B',
      description: 'Relay Tower uplink, higher risk with reinforcement trigger.',
      type: 'combat',
      encounter: {
        enemies: [
          {
            id: 'blocker-3',
            name: 'BLOCKER',
            kind: 'BLOCKER',
            maxHp: 6,
            move: 3,
            range: 5,
            damage: 1,
            ap: 2,
          },
          {
            id: 'blocker-4',
            name: 'BLOCKER',
            kind: 'BLOCKER',
            maxHp: 6,
            move: 3,
            range: 5,
            damage: 1,
            ap: 2,
          },
          {
            id: 'jammer-2',
            name: 'JAMMER',
            kind: 'JAMMER',
            maxHp: 4,
            move: 4,
            range: 4,
            damage: 1,
            ap: 2,
          },
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

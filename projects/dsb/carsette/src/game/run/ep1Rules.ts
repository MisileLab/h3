import { EncounterMode } from '../episodes/types';

export const EP1_RULES = {
  playerTurnSeconds: 30,
  stabilizerBonusSeconds: 10,
  heatThresholdExtraUpload: 4,
  heatThresholdExtractBReinforce: 5,
} as const;

export const getExtractionBaseStages = (mode: EncounterMode): number => {
  if (mode === 'EXTRACT_A') return 2;
  if (mode === 'EXTRACT_B') return 3;
  return 0;
};

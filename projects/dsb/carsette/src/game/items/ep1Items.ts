import { ItemData, ItemType } from '../types/Item';

export type ValueTier = 0 | 1 | 2 | 3;

export interface Ep1ItemData extends ItemData {
  valueTier: ValueTier;
}

const nowId = (prefix: string): string => `${prefix}-${Date.now()}-${Math.floor(performance.now())}`;

export const createScrapItem = (amount: number): Ep1ItemData => ({
  id: nowId('scrap'),
  name: `SCRAP x${amount}`,
  type: ItemType.ACCESSORY,
  shape: [[1]],
  rotation: 0,
  maxAmmo: null,
  currentAmmo: null,
  scrapValue: amount,
  valueTier: 0,
  description: 'Raw salvage. Used for construction.',
});

export const createPlateItem = (amount: number): Ep1ItemData => ({
  id: nowId('plate'),
  name: `PLATE x${amount}`,
  type: ItemType.ACCESSORY,
  shape: [[1]],
  rotation: 0,
  maxAmmo: null,
  currentAmmo: null,
  scrapValue: 0,
  valueTier: 1,
  description: 'Refined plate material.',
});

export const createBasicAttackCassette = (): Ep1ItemData => ({
  id: nowId('cassette'),
  name: 'BASIC ATTACK CASSETTE',
  type: ItemType.WEAPON,
  shape: [[1]],
  rotation: 0,
  maxAmmo: null,
  currentAmmo: null,
  scrapValue: 0,
  valueTier: 3,
  description: 'Standard issue attack cassette.',
});

export const createPatchKit = (): Ep1ItemData => ({
  id: nowId('patchkit'),
  name: 'PATCH KIT',
  type: ItemType.CONSUMABLE,
  shape: [[1]],
  rotation: 0,
  maxAmmo: null,
  currentAmmo: null,
  scrapValue: 0,
  valueTier: 2,
  description: 'Field repair consumable.',
});

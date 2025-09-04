
import Dexie, { type Table } from 'dexie';

export interface FormResponse {
  id?: number;
  email: string;
  name: string;
  number: number;
  create_time: Date;
  last_submitted_time: Date;
}

export interface Setting {
  key: string;
  value: string;
}

export class MySubClassedDexie extends Dexie {
  responses!: Table<FormResponse>; 
  settings!: Table<Setting>;

  constructor() {
    super('leadManagerDB');
    this.version(1).stores({
      responses: '++id, &[email+name+create_time], email',
      settings: '&key'
    });
  }
}

export const db = new MySubClassedDexie();

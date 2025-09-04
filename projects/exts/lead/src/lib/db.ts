
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

export interface FieldMapping {
  questionId: string;
  fieldName: string;
  originalTitle?: string;
}

export class MySubClassedDexie extends Dexie {
  responses!: Table<FormResponse>; 
  settings!: Table<Setting>;
  fieldMappings!: Table<FieldMapping>;

  constructor() {
    super('leadManagerDB');
    this.version(2).stores({
      responses: '++id, &[email+name+create_time], email',
      settings: '&key',
      fieldMappings: 'questionId'
    });
  }
}

export const db = new MySubClassedDexie();

export interface EventDate {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
}

export interface Event {
  start: EventDate;
  end: EventDate;
  title: string;
  content: string;
  color: string;
}

export interface SimpleEvent {
  year: number;
  month: number;
  day: number;
  title: string;
  content: string;
  color: string;
  org: Event;
}

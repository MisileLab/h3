export class TurnTimer {
  private secondsRemaining: number;
  private isActive: boolean;

  public constructor(startSeconds: number) {
    this.secondsRemaining = startSeconds;
    this.isActive = false;
  }

  public start(seconds: number): void {
    this.secondsRemaining = seconds;
    this.isActive = true;
  }

  public stop(): void {
    this.isActive = false;
  }

  public isRunning(): boolean {
    return this.isActive;
  }

  public tick(): number {
    if (!this.isActive) return this.secondsRemaining;
    this.secondsRemaining = Math.max(0, this.secondsRemaining - 1);
    return this.secondsRemaining;
  }

  public addSeconds(amount: number): void {
    this.secondsRemaining += amount;
  }

  public getSecondsRemaining(): number {
    return this.secondsRemaining;
  }
}

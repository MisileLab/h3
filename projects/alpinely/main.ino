void setup() {
  
}

void loop() {
  // Serial.print(analogRead(0));
  // Serial.println();
  // Serial.print(analogRead(0))
  if (analogRead(0) * 0.48828125 >= 50) {
    Serial.print("ae");
    tone(8, 311);
    delay(400);
  }
  delay(100);
}

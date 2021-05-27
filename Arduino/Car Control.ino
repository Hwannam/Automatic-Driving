void setup() {

  pinMode(2, OUTPUT);       // Motor A 방향설정1

  pinMode(4, OUTPUT);       // Motor A 방향설정2

  pinMode(6, OUTPUT);       // Motor B 방향설정1

  pinMode(7, OUTPUT);       // Motor B 방향설정2

  pinMode(8, OUTPUT);       // Motor C 방향설정1

  pinMode(9, OUTPUT);       // Motor C 방향설정2

  pinMode(12, OUTPUT);       // Motor D 방향설정1

  pinMode(13, OUTPUT);       // Motor D 방향설정2

}


void loop() {

  /*모터A설정*/

  digitalWrite(2, HIGH);     // Motor A 방향설정1

  digitalWrite(4, LOW);      // Motor A 방향설정2

  analogWrite(3, 100);       // Motor A 속도조절 (0~255)

  /*모터B설정*/

  digitalWrite(6, LOW);      // Motor B 방향설정1

  digitalWrite(7, HIGH);     // Motor B 방향설정2

  analogWrite(5, 100);        // Motor B 속도조절 (0~255)

  /*모터C설정*/

  digitalWrite(8, LOW);      // Motor B 방향설정1

  digitalWrite(9, HIGH);     // Motor B 방향설정2

  analogWrite(10, 100);        // Motor B 속도조절 (0~255)

  /*모터D설정*/

  digitalWrite(12, LOW);      // Motor B 방향설정1

  digitalWrite(13, HIGH);     // Motor B 방향설정2

  analogWrite(11, 100);        // Motor B 속도조절 (0~255)

  delay(3000);                   // 3초 유지

}

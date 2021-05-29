void setup() {

  pinMode(2, OUTPUT);       // Motor A 방향설정1

  pinMode(4, OUTPUT);       // Motor A 방향설정2

  pinMode(6, OUTPUT);       // Motor B 방향설정1

  pinMode(7, OUTPUT);       // Motor B 방향설정2

  pinMode(8, OUTPUT);       // Motor C 방향설정1

  pinMode(9, OUTPUT);       // Motor C 방향설정2

  pinMode(12, OUTPUT);       // Motor D 방향설정1

  pinMode(13, OUTPUT);       // Motor D 방향설정2

  Serial.begin(9600);

}


void loop() {

  /*모터A설정*/
  digitalWrite(2, HIGH);     // Motor A 방향설정1

  digitalWrite(4, LOW);      // Motor A 방향설정2

  /*모터B설정*/
  digitalWrite(6, LOW);      // Motor B 방향설정1

  digitalWrite(7, HIGH);     // Motor B 방향설정2

  /*모터C설정*/

  digitalWrite(8, LOW);      // Motor C 방향설정1

  digitalWrite(9, HIGH);     // Motor C 방향설정2

  /*모터D설정*/

  digitalWrite(12, LOW);      // Motor D 방향설정1

  digitalWrite(13, HIGH);     // Motor D 방향설정2

  char s = Serial.read();

  if(s == '1') // 100km
  {
    analogWrite(3, 100);
    analogWrite(5, 100);
    analogWrite(10, 100);
    analogWrite(11, 100); 
  }

  if(s == '6') // 60km
  {
    analogWrite(3, 60);
    analogWrite(5, 60);
    analogWrite(10, 60);
    analogWrite(11, 60); 
  }

  if(s == 's' || s == 'e' || s == 'o') // Stop or Red Light or Obstacle
  {
    analogWrite(3, 0);
    analogWrite(5, 0);
    analogWrite(10, 0);
    analogWrite(11, 0); 
  }

  if(s == 'r') // Turn Right 
  {
    analogWrite(3, 70);
    analogWrite(5, 100);
    analogWrite(10, 100);
    analogWrite(11, 70); 
  }

   if(s == 'l') // Turn Left 
  {
    analogWrite(3, 100);
    analogWrite(5, 70);
    analogWrite(10, 70);
    analogWrite(11, 100); 
  }
// Lane Detection 각도에 따른 방향 
}

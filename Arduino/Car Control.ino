#define TRIG A1 //TRIG 핀 설정 (초음파 보내는 핀)

#define ECHO A0 //ECHO 핀 설정 (초음파 받는 핀)


int distance;
int s;
char ch;
char state;
int dm;

void setup() {

  pinMode(2, OUTPUT);       // Motor A 방향설정1
  pinMode(4, OUTPUT);       // Motor A 방향설정2
  pinMode(6, OUTPUT);       // Motor B 방향설정1
  pinMode(7, OUTPUT);       // Motor B 방향설정2
  pinMode(8, OUTPUT);       // Motor C 방향설정1
  pinMode(9, OUTPUT);       // Motor C 방향설정2
  pinMode(12, OUTPUT);       // Motor D 방향설정1
  pinMode(13, OUTPUT);       // Motor D 방향설정2
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  Serial.begin(9600);
  dm =0;
}

int kk()
{
   long duration, distance;

  digitalWrite(TRIG, LOW);

  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);

  delayMicroseconds(10);

  digitalWrite(TRIG, LOW);

  duration = pulseIn (ECHO, HIGH); //물체에 반사되어돌아온 초음파의 시간을 변수에 저장합니다.

  //34000*초음파가 물체로 부터 반사되어 돌아오는시간 /1000000 / 2(왕복값이아니라 편도값이기때문에 나누기2를 해줍니다.)

 //초음파센서의 거리값이 위 계산값과 동일하게 Cm로 환산되는 계산공식 입니다. 수식이 간단해지도록 적용했습니다.

  distance = duration * 17 / 1000; 

  Serial.println(distance);
  if(distance<1000) return distance;
}


void loop() {
  /*모터A설정*/
  digitalWrite(2, HIGH);     // Motor A 방향설정1
  digitalWrite(4, LOW);      // Motor A 방향설정2

  /*모터B설정*/
  digitalWrite(6, HIGH);      // Motor B 방향설정1
  digitalWrite(7, LOW);     // Motor B 방향설정2

  /*모터C설정*/

  digitalWrite(8, LOW);      // Motor C 방향설정1
  digitalWrite(9, HIGH);     // Motor C 방향설정2

  /*모터D설정*/

  digitalWrite(12, HIGH);      // Motor D 방향설정1
  digitalWrite(13, LOW);     // Motor D 방향설정2

/*--------------------------------------------------------------*/

  
  if(Serial.available()){
    state = Serial.read();
  }
  
  //신호등
  if(state=='b')
  { analogWrite(3, 50);
    analogWrite(5, 50);
    analogWrite(10, 50);
    analogWrite(11, 50); 
    dm=1;
  }
  if(state=='r') {
    analogWrite(3, 0);
    analogWrite(5, 0);
    analogWrite(10, 0);
    analogWrite(11, 0); 
    dm=0;
  }

  //표지판
  if(state=='s') //stop
   {
    analogWrite(3, 0);
    analogWrite(5, 0);
    analogWrite(10, 0);
    analogWrite(11, 0); 
    dm=0;
  }
  
 // if(state=='t') //20km
 //  {
 //   analogWrite(3, 50);
 //   analogWrite(5, 50);
//    analogWrite(10, 50);
//    analogWrite(11, 50); 
 // }
  

  //차선    
  if((state=='0')*dm){
      analogWrite(3, 50);
    analogWrite(5, 255);
    analogWrite(10, 220);
    analogWrite(11, 50);

  }
  if((state=='1')*dm){
       analogWrite(3, 50);
    analogWrite(5, 130);
    analogWrite(10, 130);
    analogWrite(11, 50);
   
  }
  if((state=='2')*dm){
       analogWrite(3, 50);
    analogWrite(5, 110);
    analogWrite(10, 110);
    analogWrite(11, 50);
      
  }
   if((state=='3')*dm){
       analogWrite(3, 100);
    analogWrite(5, 10);
    analogWrite(10, 10);
    analogWrite(11, 100);
     
  }
  if((state=='4')*dm){
       analogWrite(3, 50);
    analogWrite(5, 100);
    analogWrite(10,100);
    analogWrite(11, 50);
    
  }
  if((state=='5')*dm){ //5 직진
     analogWrite(3, 70);
    analogWrite(5, 70);
    analogWrite(10, 70);
    analogWrite(11, 70);
  }
  if((state=='6')*dm){
       analogWrite(3, 100);
    analogWrite(5, 50);
    analogWrite(10, 50);
    analogWrite(11, 100);
  
  }
  if((state=='7')*dm){
      analogWrite(3, 100);
    analogWrite(5, 50);
    analogWrite(10, 50);
    analogWrite(11, 100);
   
  }
  if((state=='8')*dm){
    analogWrite(3, 220);
    analogWrite(5, 50);
    analogWrite(10, 50);
    analogWrite(11, 220);
    
  } 

 

distance = kk();
  while(distance <=20)
  {
    analogWrite(3, 0);
    analogWrite(5, 0);
    analogWrite(10, 0);
    analogWrite(11, 0);
    distance= kk();
  }
  
}

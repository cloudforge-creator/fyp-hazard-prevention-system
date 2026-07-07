/*
  Predictive AI Hazard Prevention System
  Arduino Sketch - ALL sensors in one file
  Upload THIS FILE ONLY to Arduino

  PIN CONNECTIONS:
  MQ-2 Gas Sensor  : AO->A0, VCC->5V, GND->GND
  DHT22 Temp       : DATA->Pin2, VCC->3.3V, GND->GND
  Flame Sensor     : DO->Pin4, VCC->5V, GND->GND
  Relay Module     : IN->Pin7, VCC->5V, GND->GND
*/

#include "DHT.h"

#define GAS_PIN    A0
#define DHT_PIN    2
#define FLAME_PIN  4
#define RELAY_PIN  7
#define DHT_TYPE   DHT22

DHT dht(DHT_PIN, DHT_TYPE);

// CALIBRATION - adjust CLEAN_AIR_VALUE after reading sensor 
// in clean air for 5 minutes via Serial Monitor
#define CLEAN_AIR_VALUE 400
#define MQ2_RL          10.0

bool relayActive = false;

void setup() {
  Serial.begin(9600);
  pinMode(FLAME_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  dht.begin();
  delay(2000);
  Serial.println("SYSTEM:READY");
}

void loop() {
  int   gasRaw  = analogRead(GAS_PIN);
  float voltage = gasRaw * (5.0 / 1023.0);
  float rs      = ((5.0 - voltage) / voltage) * MQ2_RL;
  float ro      = (((5.0 - (CLEAN_AIR_VALUE*5.0/1023.0)) /
                    (CLEAN_AIR_VALUE*5.0/1023.0)) * MQ2_RL) / 9.8;
  float ratio   = rs / ro;

  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();
  if (isnan(temperature)) temperature = -999;
  if (isnan(humidity))    humidity    = -999;

  int flameDetected = (digitalRead(FLAME_PIN) == LOW) ? 1 : 0;

  Serial.print("GAS_RAW:");    Serial.println(gasRaw);
  Serial.print("GAS_RATIO:");  Serial.println(ratio, 4);
  Serial.print("TEMP:");       Serial.println(temperature, 2);
  Serial.print("HUM:");        Serial.println(humidity, 2);
  Serial.print("FLAME:");      Serial.println(flameDetected);
  Serial.println("---END---");

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "RELAY:ON") {
      digitalWrite(RELAY_PIN, HIGH);
      relayActive = true;
    } else if (cmd == "RELAY:OFF") {
      digitalWrite(RELAY_PIN, LOW);
      relayActive = false;
    }
  }
  delay(500);
}

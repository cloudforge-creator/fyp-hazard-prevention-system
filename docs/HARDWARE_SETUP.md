# Hardware Setup and Bring-Up Checklist

This guide is a preparation checklist. Pin assignments and voltage levels must be confirmed against the exact board and module revisions before wiring.

## 1. Prepare safely

- [ ] Identify the exact controller board (Arduino model or Raspberry Pi model) and record its operating voltage.
- [ ] Obtain the wiring diagrams/datasheets for the MQ gas sensor, DHT11/DHT22, flame sensor, camera, buzzer and relay module.
- [ ] Use a current-limited bench supply where possible.
- [ ] Disconnect mains and other hazardous loads during initial testing. Test the relay with a low-voltage lamp or meter first.
- [ ] Do not connect a 5 V signal directly to a 3.3 V-only GPIO input.
- [ ] Keep sensor electronics away from open flames, smoke, combustible gas and high-temperature experiments.

## 2. Planned signal map (verify before wiring)

| Component | Controller interface | Bring-up check |
|---|---|---|
| MQ-2 gas sensor | Analog output / ADC input | Verify ADC range, module voltage, warm-up and calibration requirements |
| DHT11 or DHT22 | Digital data pin | Confirm library, pull-up requirements and sampling interval |
| Flame sensor | Digital output; analog output only if supported | Confirm active-high/active-low behavior with a meter and safe test source |
| Camera | USB or camera interface | Confirm OS device enumeration and camera permissions |
| Buzzer/siren | GPIO through a suitable driver | Check current draw; do not power a high-current siren directly from a GPIO |
| Relay module | Digital output through compatible driver/module | Verify trigger polarity, coil supply and isolation; test without mains connected |

**Do not use guessed pin numbers.** Fill in the actual board pins only after the exact hardware versions are known.

## 3. Electrical checks before connecting sensors

- [ ] Confirm common ground where required by the interface design.
- [ ] Verify each module's supply voltage and maximum current.
- [ ] Check analog input voltage is within the controller's permitted range.
- [ ] Verify the relay module's input compatibility and default state at boot.
- [ ] Add suitable power regulation and protection; do not power motors or sirens from the controller's logic rail.
- [ ] Ensure the system has a safe, manually accessible way to isolate the load.

## 4. Software bring-up sequence

1. Create and activate the project virtual environment; install requirements.txt.
2. Start with python src/main.py --sim and confirm the dashboard and event log work without hardware.
3. Connect one sensor at a time, with the relay load still disconnected.
4. Select the explicit serial port, for example --port COM4 on Windows, after confirming it in Device Manager. Never assume COM4 is correct.
5. Verify serial messages include valid gas ratio/raw ADC, temperature, humidity and flame status. Record missing or out-of-range values rather than silently replacing them.
6. Compare each displayed reading with a suitable reference instrument.
7. Test camera detection using labeled fire and non-fire examples; measure false positives and false negatives.
8. Test buzzer and relay behavior using a low-voltage test load, including power-up, disconnect, reset and software shutdown.
9. Only after bench tests pass, conduct a documented supervised integration test.

## 5. Calibration and acceptance record

Record the following for each test:

- Date/time, tester and hardware model/revision.
- Sensor part number, supply voltage, warm-up time and calibration method.
- Reference reading and system reading, including units.
- Normal, warning and critical test cases, repetitions and observed response latency.
- False alarms, missed detections, sensor disconnect behavior and recovery behavior.
- Relay default state, trigger polarity and reset behavior.
- Dataset/model version and software commit SHA.

Do not claim accuracy, latency, or 100% alert activation without a documented test procedure and results. A sensor simulator is not evidence of hardware accuracy.

## 6. Stop conditions

Disconnect power and stop testing if a module overheats, wiring or voltage is uncertain, readings are unstable, the relay behaves unexpectedly, or the system fails to enter a known safe state. Never perform live mains switching during initial development.

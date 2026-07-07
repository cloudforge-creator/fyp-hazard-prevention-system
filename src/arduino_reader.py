"""
arduino_reader.py
Handles all serial communication with Arduino.
Runs in simulation mode when no hardware is connected.
"""
import serial
import serial.tools.list_ports
import time
import threading
import math


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if any(x in desc for x in ['arduino','ch340','usb serial','usbmodem','cp210']):
            return port.device
    if ports:
        return ports[0].device
    return None


class ArduinoReader:
    def __init__(self, port=None, baud=9600, simulation=False):
        self.simulation = simulation
        self.connected  = False
        self.ser        = None
        self._lock      = threading.Lock()

        if simulation:
            print("[ArduinoReader] SIMULATION mode - no hardware needed")
            return

        port = port or find_arduino_port()
        if not port:
            print("[ArduinoReader] No Arduino found - switching to SIMULATION")
            self.simulation = True
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=2)
            time.sleep(2)
            self.ser.flushInput()
            self.connected = True
            print(f"[ArduinoReader] Connected on {port}")
        except Exception as e:
            print(f"[ArduinoReader] Connection failed ({e}) - SIMULATION mode")
            self.simulation = True

    def read_sensors(self):
        if self.simulation:
            return self._simulate()
        data = {'gas_raw': None, 'gas_ratio': None,
                'temp': None, 'humidity': None, 'flame': 0}
        try:
            start = time.time()
            while time.time() - start < 3:
                raw  = self.ser.readline()
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                if line == '---END---':
                    break
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                k = key.strip(); v = val.strip()
                try:
                    if   k == 'GAS_RAW':   data['gas_raw']   = int(float(v))
                    elif k == 'GAS_RATIO': data['gas_ratio'] = float(v)
                    elif k == 'TEMP':      data['temp']      = float(v) if float(v) != -999 else None
                    elif k == 'HUM':       data['humidity']  = float(v) if float(v) != -999 else None
                    elif k == 'FLAME':     data['flame']     = int(v)
                except ValueError:
                    pass
        except Exception as e:
            print(f"[ArduinoReader] Read error: {e}")
            return self._simulate()
        return data

    def activate_relay(self):
        if self.simulation:
            print("[SIM] *** RELAY ON - machine power would be cut ***"); return
        try:
            with self._lock:
                self.ser.write(b'RELAY:ON\n')
            print("[ArduinoReader] Relay ON sent")
        except Exception as e:
            print(f"[ArduinoReader] Relay error: {e}")

    def deactivate_relay(self):
        if self.simulation:
            print("[SIM] Relay OFF"); return
        try:
            with self._lock:
                self.ser.write(b'RELAY:OFF\n')
        except Exception as e:
            print(f"[ArduinoReader] Relay off error: {e}")

    def _simulate(self):
        t = time.time()
        return {
            'gas_raw':   int(350 + 50 * math.sin(t / 10)),
            'gas_ratio': round(1.2 + 0.3 * math.sin(t / 10), 4),
            'temp':      round(28.0 + 2.0 * math.sin(t / 30), 2),
            'humidity':  round(55.0 + 5.0 * math.cos(t / 20), 2),
            'flame':     0
        }

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

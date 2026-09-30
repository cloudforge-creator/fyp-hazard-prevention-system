"""
main.py
Entry point for the entire Predictive AI Hazard Prevention System.
"""
import sys
import os
import threading
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arduino_reader import ArduinoReader
from src.gas_model import predict_gas
from src.lstm_model import predict_anomaly
from src.cnn_model import predict_fire, get_jpeg_frame
from src.decision_engine import DecisionEngine
from src.firebase_handler import FirebaseHandler
from dashboard.app import run_dashboard, update_live_data, add_event


def parse_args():
    parser = argparse.ArgumentParser(description='FYP Hazard Prevention System')
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
    parser.add_argument('--port', type=str, default=None,
                        help='Arduino serial port (e.g. COM4 or /dev/ttyUSB0)')
    parser.add_argument('--no-firebase', action='store_true',
                        help='Disable Firebase (simulation only)')
    parser.add_argument('--web-port', type=int, default=5000,
                        help='Flask dashboard port (default 5000)')
    return parser.parse_args()


def sensor_loop(arduino, engine, firebase):
    print("[Main] Sensor loop started")
    cycle = 0

    while True:
        cycle += 1
        loop_start = time.time()
        try:
            sensors = arduino.read_sensors()
            gas_ratio = sensors.get('gas_ratio')
            gas_raw = sensors.get('gas_raw', 0)
            temp = sensors.get('temp')
            humidity = sensors.get('humidity')
            flame = sensors.get('flame', 0)

            if gas_ratio is None and temp is None:
                print(f"[Main] Cycle {cycle}: No sensor data - skipping")
                time.sleep(0.5)
                continue

            gas_label, gas_risk = predict_gas(gas_ratio or 5.0, gas_raw or 0)
            temp_anomaly, temp_error, temp_predicted = predict_anomaly(temp or 25.0)
            fire_prob, _ = predict_fire()

            result = engine.evaluate(
                gas_label=gas_label,
                gas_risk=gas_risk,
                temp_anomaly=temp_anomaly,
                temp_error=temp_error,
                fire_prob=fire_prob,
                flame_signal=flame
            )

            update_live_data({
                'gas': gas_label,
                'gas_risk': gas_risk,
                'temperature': temp,
                'humidity': humidity,
                'fire_prob': fire_prob,
                'temp_anomaly': temp_anomaly,
                'temp_error': temp_error,
                'risk_level': result['level'],
                'risk_score': result['score'],
                'method': result['method'],
                'cycle': cycle,
                'updated_at': time.strftime('%H:%M:%S')
            })

            if cycle % 10 == 0:
                firebase.update_live_sensors(result)

            if result['level'] != 'SAFE':
                add_event(result)

            level_indicator = {'SAFE': '✓', 'WARNING': '⚠', 'CRITICAL': '✗'}.get(
                result['level'], '?')
            print(f"[{level_indicator}] Cycle {cycle:04d} | "
                  f"Gas:{gas_label:<12} Risk:{gas_risk:.2f} | "
                  f"Temp:{temp or 0:.1f}°C Anom:{temp_anomaly} | "
                  f"Fire:{fire_prob:.0%} | "
                  f"Level:{result['level']} Score:{result['score']:.3f}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[Main] Cycle {cycle} error: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - loop_start
        time.sleep(max(0, 0.5 - elapsed))


def main():
    args = parse_args()
    sim_mode = args.sim

    print("=" * 60)
    print("  Predictive AI Hazard Prevention System")
    print("  Final Year Project")
    print("=" * 60)
    print(f"  Mode: {'SIMULATION' if sim_mode else 'HARDWARE'}")
    print(f"  Dashboard: http://localhost:{args.web_port}")
    print("=" * 60)

    print("\n[Main] Initialising components...")
    arduino = ArduinoReader(port=args.port, simulation=sim_mode)
    firebase = FirebaseHandler(simulation=(sim_mode or args.no_firebase))
    engine = DecisionEngine(firebase_handler=firebase, arduino_reader=arduino)

    print("[Main] All components initialised")
    print("[Main] Models will load on first prediction call")
    print("[Main] Starting sensor loop...")

    loop_thread = threading.Thread(
        target=sensor_loop, args=(arduino, engine, firebase),
        daemon=True, name='SensorLoop')
    loop_thread.start()

    time.sleep(1.5)

    try:
        run_dashboard(
            get_jpeg_fn=get_jpeg_frame,
            firebase=firebase,
            reset_relay_fn=engine.reset_relay,
            port=args.web_port
        )
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        arduino.deactivate_relay()
        arduino.close()
        print("[Main] System stopped safely")


if __name__ == '__main__':
    main()

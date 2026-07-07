"""
main.py
Entry point for the entire Predictive AI Hazard Prevention System.
Run this file: python src/main.py

Modes:
  python src/main.py              -> auto-detect hardware
  python src/main.py --sim        -> full simulation (no hardware needed)
  python src/main.py --port COM4  -> specify Arduino port manually
"""
import sys
import os
import threading
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arduino_reader  import ArduinoReader
from src.gas_model       import predict_gas
from src.lstm_model      import predict_anomaly
from src.cnn_model       import predict_fire, get_jpeg_frame
from src.decision_engine import DecisionEngine
from src.firebase_handler import FirebaseHandler
from dashboard.app       import (run_dashboard, update_live_data,
                                  add_event)


def parse_args():
    parser = argparse.ArgumentParser(description='FYP Hazard Prevention System')
    parser.add_argument('--sim',    action='store_true',
                        help='Run in simulation mode (no hardware)')
    parser.add_argument('--port',   type=str, default=None,
                        help='Arduino serial port (e.g. COM4 or /dev/ttyUSB0)')
    parser.add_argument('--no-firebase', action='store_true',
                        help='Disable Firebase (simulation only)')
    parser.add_argument('--web-port', type=int, default=5000,
                        help='Flask dashboard port (default 5000)')
    return parser.parse_args()


def sensor_loop(arduino: ArduinoReader,
                engine:  DecisionEngine,
                firebase: FirebaseHandler):
    """
    Main continuous loop:
    1. Read sensors from Arduino (or simulation)
    2. Run all three AI models
    3. Pass outputs to Decision Engine
    4. Update dashboard live data
    5. Sleep 500ms and repeat
    """
    print("[Main] Sensor loop started")
    cycle = 0

    while True:
        cycle += 1
        loop_start = time.time()

        try:
            # ---- Step 1: Read Hardware ----
            sensors = arduino.read_sensors()

            gas_ratio = sensors.get('gas_ratio')
            gas_raw   = sensors.get('gas_raw', 0)
            temp      = sensors.get('temp')
            humidity  = sensors.get('humidity')
            flame     = sensors.get('flame', 0)

            # Skip cycle if critical readings are missing
            if gas_ratio is None and temp is None:
                print(f"[Main] Cycle {cycle}: No sensor data - skipping")
                time.sleep(0.5)
                continue

            # ---- Step 2: Run Three AI Models ----
            # Model 1: Random Forest gas classification
            gas_label, gas_risk = predict_gas(
                gas_ratio or 5.0, gas_raw or 0)

            # Model 2: LSTM temperature anomaly
            temp_anomaly, temp_error, temp_predicted = predict_anomaly(
                temp or 25.0)

            # Model 3: CNN fire detection
            fire_prob, _ = predict_fire()

            # ---- Step 3: Decision Engine fusion ----
            result = engine.evaluate(
                gas_label    = gas_label,
                gas_risk     = gas_risk,
                temp_anomaly = temp_anomaly,
                temp_error   = temp_error,
                fire_prob    = fire_prob,
                flame_signal = flame
            )

            # ---- Step 4: Update dashboard live data ----
            live = {
                'gas':         gas_label,
                'gas_risk':    gas_risk,
                'temperature': temp,
                'humidity':    humidity,
                'fire_prob':   fire_prob,
                'temp_anomaly':temp_anomaly,
                'temp_error':  temp_error,
                'risk_level':  result['level'],
                'risk_score':  result['score'],
                'method':      result['method'],
                'cycle':       cycle,
                'updated_at':  time.strftime('%H:%M:%S')
            }
            update_live_data(live)

            # Update Firebase live sensors every 10 cycles (~5 seconds)
            if cycle % 10 == 0:
                firebase.update_live_sensors(result)

            # Log event to dashboard history if not safe
            if result['level'] != 'SAFE':
                add_event(result)

            # Console output every cycle
            level_indicator = {
                'SAFE':     '✓',
                'WARNING':  '⚠',
                'CRITICAL': '✗'
            }.get(result['level'], '?')

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

        # Maintain 500ms cycle time regardless of processing duration
        elapsed = time.time() - loop_start
        sleep_time = max(0, 0.5 - elapsed)
        time.sleep(sleep_time)


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

    # ---- Initialise all components ----
    print("\n[Main] Initialising components...")

    arduino  = ArduinoReader(port=args.port, simulation=sim_mode)
    firebase = FirebaseHandler(simulation=(sim_mode or args.no_firebase))
    engine   = DecisionEngine(firebase_handler=firebase,
                               arduino_reader=arduino)

    print("[Main] All components initialised")
    print("[Main] Models will load on first prediction call")
    print(f"[Main] Starting sensor loop...")

    # ---- Run sensor loop in background thread ----
    loop_thread = threading.Thread(
        target=sensor_loop,
        args=(arduino, engine, firebase),
        daemon=True,
        name='SensorLoop'
    )
    loop_thread.start()

    # Wait for first cycle before starting dashboard
    time.sleep(1.5)

    # ---- Run Flask dashboard on main thread ----
    # (must be main thread for signal handling)
    try:
        run_dashboard(
            get_jpeg_fn=get_jpeg_frame,
            firebase=firebase,
            port=args.web_port
        )
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        arduino.deactivate_relay()  # safety: turn off relay on exit
        arduino.close()
        print("[Main] System stopped safely")


if __name__ == '__main__':
    main()

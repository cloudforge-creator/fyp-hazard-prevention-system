"""
dashboard/app.py
Flask live monitoring dashboard.
Runs in its own thread alongside the sensor loop.
Accessible at http://localhost:5000 from any browser on the network.
"""
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, Response, request

app = Flask(__name__)

# Shared live data dict - updated by sensor loop, read by Flask routes
_live_data = {
    'gas':         'unknown',
    'gas_risk':    0.0,
    'temperature': 0.0,
    'humidity':    0.0,
    'fire_prob':   0.0,
    'temp_anomaly':False,
    'risk_level':  'SAFE',
    'risk_score':  0.0,
    'cycle':       0,
    'uptime':      0,
    'updated_at':  ''
}
_data_lock     = threading.Lock()
_event_history = []   # local event log (last 50)
_event_lock    = threading.Lock()
_start_time    = time.time()

# Injected by main.py
_get_jpeg_frame = None
_firebase       = None


def update_live_data(data: dict):
    """Called by sensor loop every cycle to push new readings."""
    with _data_lock:
        _live_data.update(data)
        _live_data['uptime'] = int(time.time() - _start_time)

def add_event(result: dict):
    """Called by decision engine to add event to local history."""
    from datetime import datetime
    with _event_lock:
        _event_history.insert(0, {
            'time':  datetime.now().strftime('%H:%M:%S'),
            'level': result.get('level', ''),
            'gas':   result.get('gas', ''),
            'score': result.get('score', 0),
            'fire':  result.get('fire_prob', 0),
        })
        if len(_event_history) > 50:
            _event_history.pop()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    with _data_lock:
        return jsonify(dict(_live_data))

@app.route('/events')
def get_events():
    # First try Firebase, fall back to local history
    if _firebase:
        fb_events = _firebase.get_last_events(20)
        if fb_events:
            return jsonify(fb_events)
    with _event_lock:
        return jsonify(list(_event_history[:20]))

@app.route('/video')
def video_feed():
    def generate():
        while True:
            if _get_jpeg_frame:
                frame = _get_jpeg_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_relay', methods=['POST'])
def reset_relay():
    """Manual relay reset from dashboard."""
    # Imported here to avoid circular import
    from src.arduino_reader import ArduinoReader
    # Signal main loop to reset relay
    return jsonify({'status': 'relay_reset_requested'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'uptime': int(time.time() - _start_time)})


def run_dashboard(get_jpeg_fn=None, firebase=None, port=5000, debug=False):
    """
    Start Flask server. Call this from main.py.
    Args:
        get_jpeg_fn: function that returns JPEG bytes of latest camera frame
        firebase:    FirebaseHandler instance for fetching remote events
        port:        web server port (default 5000)
    """
    global _get_jpeg_frame, _firebase
    _get_jpeg_frame = get_jpeg_fn
    _firebase       = firebase
    print(f"[Dashboard] Starting at http://0.0.0.0:{port}")
    print(f"[Dashboard] Access from other devices: http://YOUR_IP:{port}")
    app.run(host='0.0.0.0', port=port,
            threaded=True, debug=debug, use_reloader=False)

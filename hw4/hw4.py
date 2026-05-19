import os
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")

import atexit
from flask import Flask, redirect, url_for, jsonify
from gpiozero import LED

app = Flask(__name__)

LED_PINS = [23, 24]
leds = {pin: LED(pin) for pin in LED_PINS}


def get_state(pin: int) -> str:
    return "on" if leds[pin].is_lit else "off"


def set_led(pin: int, action: str):
    if pin not in leds:
        return {"ok": False, "error": f"GPIO {pin} is not configured"}, 404

    if action == "on":
        leds[pin].on()
    elif action == "off":
        leds[pin].off()
    elif action == "toggle":
        leds[pin].toggle()
    else:
        return {"ok": False, "error": "action must be on, off, or toggle"}, 400

    return {"ok": True, "pin": pin, "state": get_state(pin)}, 200


@app.route("/")
def index():
    html = """
    <h1>IoT26 HW04 - Raspberry Pi 5 Flask LED Control</h1>
    <p>GPIO 23 and GPIO 24 LEDs can be controlled from this page.</p>
    """

    for pin in LED_PINS:
        state = get_state(pin).upper()
        html += f"""
        <div style="border:1px solid #aaa; padding:12px; margin:12px; width:340px;">
          <h2>GPIO {pin}: {state}</h2>
          <a href="/led/{pin}/on">
            <button style="font-size:18px;">ON</button>
          </a>
          <a href="/led/{pin}/off">
            <button style="font-size:18px;">OFF</button>
          </a>
          <a href="/led/{pin}/toggle">
            <button style="font-size:18px;">TOGGLE</button>
          </a>
        </div>
        """

    html += """
    <p>API examples:</p>
    <pre>
/api/led/23/on
/api/led/23/off
/api/led/24/on
/api/led/24/off
/api/status
    </pre>
    """
    return html


@app.route("/led/<int:pin>/<action>")
def web_control(pin, action):
    set_led(pin, action)
    return redirect(url_for("index"))


@app.route("/api/led/<int:pin>/<action>")
def api_control(pin, action):
    body, status = set_led(pin, action)
    return jsonify(body), status


@app.route("/api/status")
def api_status():
    return jsonify({pin: get_state(pin) for pin in LED_PINS})


def cleanup():
    for led in leds.values():
        led.off()
        led.close()


atexit.register(cleanup)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

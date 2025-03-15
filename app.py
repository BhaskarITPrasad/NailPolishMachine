from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import requests
import cv2
import numpy as np
import time
import subprocess
import os
import socket
import ipaddress
import threading

app = Flask(__name__)

camera_url = None
esp32_ip = None
streaming = False
latest_frame = None
active_devices = []  # List of (IP, Hostname)
scanning = False
scan_thread = None

# -----------------------------
# Helper Functions
# -----------------------------

def gen_frames(url):
    """Generate frames from ESP32 camera stream."""
    global streaming, latest_frame
    while streaming:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                np_arr = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    latest_frame = frame
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error fetching frame: {e}")
            time.sleep(1)


def get_local_ip():
    """Get the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_subnet(ip):
    """Extract subnet from IP."""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def ping(ip):
    """Ping an IP to check if it's active."""
    try:
        subprocess.check_output(["ping", "-c", "1", "-W", "1", ip], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def assign_device_names(devices):
    """
    Assign 'Nail Printer Camera' and 'Plotter' names to two consecutive unknown devices.
    Only those devices will be kept in the final list.
    """
    unknown_devices = [device for device in devices if device[1] == "Unknown Device"]
    unknown_devices.sort(key=lambda x: list(map(int, x[0].split('.'))))

    # Default empty list if no matches found
    final_devices = []

    # Find consecutive unknown devices
    for i in range(len(unknown_devices) - 1):
        current_ip = list(map(int, unknown_devices[i][0].split('.')))
        next_ip = list(map(int, unknown_devices[i + 1][0].split('.')))

        if current_ip[:-1] == next_ip[:-1] and next_ip[-1] - current_ip[-1] == 1:
            final_devices = [
                (unknown_devices[i][0], "Nail Printer Camera"),
                (unknown_devices[i + 1][0], "Plotter")
            ]
            break

    return final_devices


def get_hostname(ip):
    """Get hostname or return 'Unknown Device'."""
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return "Unknown Device"


def scan_network(subnet):
    """Scan the network for devices."""
    global active_devices, scanning
    scanning = True
    active_devices.clear()

    print(f"🔍 Scanning network: {subnet}")
    threads = []

    def worker(ip):
        if ping(ip):
            hostname = get_hostname(ip)
            print(f"✅ Found: {ip} ({hostname})")
            active_devices.append((ip, hostname))

    for ip in ipaddress.IPv4Network(subnet, strict=False).hosts():
        thread = threading.Thread(target=worker, args=(str(ip),))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Keep only Nail Printer Camera & Plotter
    active_devices[:] = assign_device_names(active_devices)
    scanning = False
    print("✅ Scanning complete. Devices:", active_devices)


# -----------------------------
# Flask Routes
# -----------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page: Shows dropdown with devices and controls."""
    global camera_url, esp32_ip, streaming

    if request.method == 'POST':
        ip_address = request.form.get('ip_address').strip()
        camera_url = f'http://{ip_address}/capture'
        esp32_ip = ip_address
        streaming = False

    return render_template('index.html',
                           camera_url=camera_url,
                           streaming=streaming,
                           active_devices=active_devices,
                           scanning=scanning)

@app.route('/capture_image')
def capture_image():
    """Capture the current frame, process it, and show processed image."""
    global latest_frame
    if latest_frame is not None:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, latest_frame)
        print(f"📸 Image captured and saved as {image_path}")

        # Process the image to generate predictions.txt
        subprocess.run(["python", "nail_image_processing.py", image_path])
        return redirect(url_for('show_processed_image'))
    return "❌ No frame available to capture. Please start the stream first."

@app.route('/show_processed_image')
def show_processed_image():
    """Display the processed image with upload and view buttons."""
    return render_template('processed_image.html',
                           image_url=url_for('static', filename='annotated_output.jpg'))

from flask import Flask, render_template, request, jsonify, url_for
import cv2
import os

# -----------------------------
# Route: Display Color Picker Page
# -----------------------------
@app.route('/color_picker')
def color_picker():
    """Render a page where users can click on the image to pick colors."""
    return render_template('color_picker.html')  # Create this template with similar structure to processed_image.html

# -----------------------------
# Route: Get CMYK Values from Clicked Coordinates
# -----------------------------
@app.route('/get_cmyk', methods=['POST'])
def get_cmyk():
    data = request.get_json()
    x, y = data['x'], data['y']

    # Load the image
    image_path = os.path.join('static', 'image.png')
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Image not found.'}), 404

    # Get RGB values
    b, g, r = image[y, x].tolist()

    # Convert RGB to CMYK
    def rgb_to_cmyk(r, g, b):
        if (r, g, b) == (0, 0, 0):
            return 0, 0, 0, 1
        c = 1 - r / 255
        m = 1 - g / 255
        y = 1 - b / 255
        k = min(c, m, y)
        c = (c - k) / (1 - k) if (1 - k) != 0 else 0
        m = (m - k) / (1 - k) if (1 - k) != 0 else 0
        y = (y - k) / (1 - k) if (1 - k) != 0 else 0
        return c, m, y, k

    c, m, y_val, k = rgb_to_cmyk(r, g, b)

    # Save to colour.txt
    with open('colour.txt', 'w') as file:
        file.write(f"C:{c:.2f}\nM:{m:.2f}\nY:{y_val:.2f}\nK:{k:.2f}\n")

    return jsonify({'r': r, 'g': g, 'b': b, 'c': c, 'm': m, 'y': y_val, 'k': k})


@app.route('/run_pycode', methods=['POST'])
def run_pycode():
    try:
        # Execute PyCOde.py to process CMYK values and generate image.png
        subprocess.run(["python3", "PyCOde.py"], check=True)
        print("✅ PyCOde.py executed successfully.")
    except subprocess.CalledProcessError:
        print("❌ Error executing PyCOde.py.")
        return "❌ Error executing PyCOde.py.", 500

    return render_template('view_file.html', image_url=url_for('static', filename='image.png'))


@app.route('/start_scan', methods=['POST'])
def start_scan():
    """Start network scan."""
    global scan_thread, scanning
    if not scanning:
        local_ip = get_local_ip()
        subnet = get_subnet(local_ip)
        scan_thread = threading.Thread(target=scan_network, args=(subnet,), daemon=True)
        scan_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_scanning"})


@app.route('/check_scan_status', methods=['GET'])
def check_scan_status():
    """Check if scanning is complete."""
    return jsonify({"scanning": scanning})


@app.route('/start_stream')
def start_stream():
    """Start camera stream."""
    global streaming
    streaming = True
    return redirect(url_for('index'))


@app.route('/stop_stream')
def stop_stream():
    """Stop camera stream."""
    global streaming
    streaming = False
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    """Serve video stream."""
    if camera_url and streaming:
        return Response(gen_frames(camera_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No active stream. Please select a device."


if __name__ == '__main__':
    os.makedirs("templates", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

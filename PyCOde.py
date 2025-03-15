import cv2
import numpy as np
import serial

# Serial communication setup (adjust port and baudrate accordingly)
#ser = serial.Serial('/dev/ttyACM0', 9600)  # Replace 'COM3' with your Arduino port
# /dev/ttyACM0
# Load the image
image = cv2.imread('image.png')

# Function to convert RGB to CMYK
def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, 1
    # RGB [0,255] -> CMY [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.
    min_cmy = min(c, m, y)
    # CMY -> CMYK
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy
    return c, m, y, k

# Mouse click event to get the color
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = image[y, x]
        c, m, y, k = rgb_to_cmyk(r, g, b)
        
        # Send CMYK values to Arduino
        command = f"C{c:.2f}M{m:.2f}Y{y:.2f}K{k:.2f}\n"
        #ser.write(command.encode())
        
        print(f"Clicked Color (RGB): {r}, {g}, {b}")
        print(f"CMYK: {c:.2f}, {m:.2f}, {y:.2f}, {k:.2f}")
        #ack = ser.readline().decode().strip()
        #print(f"Acknowledgment from Arduino: {ack}")

# Display the image and set the mouse callback
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
ser.close()

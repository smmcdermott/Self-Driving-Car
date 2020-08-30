import socketio
from flask import Flask
import eventlet
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


# Use websocket to make live connection between socket and server and continuous connect to client
sio = socketio.Server()


# Initialize flask application
app = Flask(__name__)

speed_limit = 10

# Preprocess our images
def img_preprocess(img):
  # Crop image to remove areas the hood of car and background which are not relevant
  # Image has height of 160 and width of 320; Only keep relevant area
  img = img[60:135, :, :]

  # Convert color space to YUV (y = luminosity, uv = chrominance components) to fit nvidia model architecture format
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

  # Apply Gaussian blur to smooth image and reduce noise
  img = cv2.GaussianBlur(img, (3,3), 0)

  # Reduce size of image to fit nvidia model architecture
  img = cv2.resize(img, (200, 66))

  # Normalize image
  img = img / 255
  return img


# Set even handler to telemetry (listen to updates)
@sio.on('telemetry')

# Get image of current location from simulator, and we use this image to predict steering angle, and we use this prediction to move the car
def telemetry(sid, data):
    # Get speed of the car
    speed = float(data['speed'])
    # Read in image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # Convert image to array
    image = np.asarray(image)
    # Preprocess image in same manner as our model
    image = img_preprocess(image)
    # Convert image to 4 dimensions to get predictions
    image = np.array([image])
    # Predict steering angle of image
    steering_angle = float(model.predict(image))
    # Ensure car moves at constant speed
    throttle = 1.0 - (speed / speed_limit)
    # print steering angle, throttle, and speed
    print(f'{steering_angle} {throttle} {speed}')
    
    # Send steering angle to simulator to play action accordingly
    send_control(steering_angle, 1.0)

# Set event handler to connect to application
@sio.on('connect')

def connect(sid, environ):
    print('Connected')
    # Give simulator an initial steering angle of 0 and throttle of 0 (stationary)
    send_control(0, 0)
    
# Send the image data to the 
def send_control(steering_angle, throttle):
    # Have the car steer acording to data passed to it
    sio.emit('steer', data={'steering_angle': steering_angle.__str__(),
                            'throttle': throttle.__str__()
                            })

if __name__ == '__main__':
    # Load in the trained model
    model = load_model('model.h5', compile=False)
    app = socketio.Middleware(sio, app)
    # Have the simulation listen to anything passed to it
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

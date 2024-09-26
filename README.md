# Real-Time Object Detection and Distance Estimation using MobileNet-SSD on Raspberry Pi

This project demonstrates a real-time object detection system using a Raspberry Pi and MobileNet-SSD. It also estimates the distance between the camera and detected objects based on their known heights, making it useful for smart systems that require spatial awareness.

## Features

- Real-time object detection using the MobileNet-SSD deep learning model.
- Distance estimation from the camera to the detected objects using simple geometry based on known object heights.
- Alerts when objects come too close to the camera.

## Hardware Requirements

- Raspberry Pi 4 Model B (or equivalent)
- Raspberry Pi Camera Module (or USB webcam)
- Power supply for Raspberry Pi
- Monitor, keyboard, and mouse (for setup)
- Internet connection for installing dependencies

## Software Requirements

- **Raspbian OS**: Make sure your Raspberry Pi is running the latest Raspbian OS.
- **Python 3.x**: This project is built using Python 3.x.
- **OpenCV**: OpenCV library for image processing and object detection.
- **NumPy**: Required for handling arrays and numerical operations.
- **PiCamera**: Python interface for Raspberry Pi Camera.

## Model and Dataset

We use the **MobileNet-SSD** pre-trained model for object detection. The following files are needed:
- `deploy.prototxt`: The architecture of the MobileNet-SSD.
- `mobilenet_iter_73000.caffemodel`: The pre-trained weights of MobileNet-SSD.

These files can be downloaded from the [MobileNet-SSD Caffe Model repository](https://github.com/chuanqi305/MobileNet-SSD).

## Installation

Follow these steps to set up the project on your Raspberry Pi:

### 1. Clone the Repository
First, clone the project repository:

```bash
git clone https://github.com/yourusername/Object-Detection-using-MobilenetSSD.git
cd Object-Detection-using-MobilenetSSD
```

### 2. Install Required Dependencies
Make sure the required libraries are installed. You can install the dependencies using `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

Dependencies in `requirements.txt`:
```txt
opencv-python
numpy
picamera[array]
```

Alternatively, you can install these manually:
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-picamera python3-numpy
```

### 3. Place Model Files
Download and place the MobileNet-SSD model files in the `models/` directory:

```
Object-Detection-using-MobilenetSSD/
├── models/
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
```

### 4. Run the Object Detection Script

Once everything is set up, you can run the object detection script as follows:

```bash
python3 src/object_detection_with_picam.py
```

## Usage Instructions

After starting the script:
1. The system will initialize the Raspberry Pi camera.
2. The MobileNet-SSD model will start detecting objects in the camera's view.
3. The distance from the camera to each detected object will be estimated based on known heights of objects (e.g., a person, a bottle, etc.).
4. If an object is too close, a "!! TOO CLOSE !!" alert will be displayed on the screen.

To exit the program, press `q` in the window.

## Directory Structure

The project is organized as follows:

```
Object-Detection-using-MobilenetSSD/
│
├── models/                           # Pre-trained model files (prototxt and caffemodel)
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
│
├── src/                              # Source code directory
│   └── object_detection_with_picam.py
│
├── README.md                         # Project documentation
├── requirements.txt                  # Project dependencies
└── LICENSE                           # License for the project
```

## Known Object Heights

The script uses estimated heights (in centimeters) for known objects to calculate the distance. These are defined in the script under `KNOWN_HEIGHTS`:

```python
KNOWN_HEIGHTS = {
    "person": 170,
    "bottle": 30,
    "car": 150,
    "bus": 300,
    "chair": 100,
    "diningtable": 75,
    "tvmonitor": 60,
}
```

You can modify or add more objects to this list based on your application needs.

## Focal Length Calibration

The focal length used in the distance calculation is set as a constant (`FOCAL_LENGTH = 615`). You may need to adjust this value based on your camera's specifications for more accurate distance measurements.

## Contribution

Feel free to contribute by:
- Adding more object classes and their known heights.
- Improving distance estimation accuracy.
- Optimizing the script for performance.

This is the hardware setup we had done with raspberry-pi
![Hardware Setup](https://github.com/RanjithKumar17904/Real-Time-Object-Detection-and-Distance-Estimation-using-MobileNet-SSD-on-Raspberry-Pi/blob/main/setup%20image.jpg)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## References

- [MobileNet-SSD Caffe Model](https://github.com/chuanqi305/MobileNet-SSD)
- [OpenCV Documentation](https://docs.opencv.org/)

## ALEVision - Retail Traffic Insight Tool 🎁
This project is designed to analyze human behavior in retail environments. It enables store owners and researchers to understand customer movement patterns, dwell times in specific zones, and overall store dynamics. The system anonymizes individuals to protect privacy while providing insights into crowd behavior.

## Features 🤓

- **Movement Analysis**: Tracks how people move throughout the store.
- **Time Monitoring**: Measures the time customers spend in different zones.
- **Anonymization**: Ensures individual privacy by anonymizing users' identities.
- **Integration with YOLO**: Allows for specific queries on detected objects and individuals.
- **Multi-Camera Tracking**: Capable of tracking individuals across multiple camera feeds.
- **Bird's Eye View**: Transforms the perspective for a top-down analysis of movement patterns.
- **Model Retraining**: Saves random images to re-train the machine learning model for improved accuracy.
- **Heat Maps**: Generates heat maps to visualize high-traffic areas.

## Prerequisites 📌

Before running this system, you need to:

- Install Python 3.9 or above.
- Ensure you have the required hardware to process video feeds in real-time.
- Set up the necessary camera infrastructure as per the system's requirements.

## Installation

```bash
git clone https://github.com/your-repository/human-behavior-analysis.git
cd human-behavior-analysis
pip install git+https://github.com/AdonaiVera/supervision.git@add_time_byte_tracking
pip install ultralytics
```

## Usage 📌

To start analyzing a video stream:

```python
from behavior_analysis import process_video

process_video(
    source_weights_path="path/to/yolo-weights",
    stream_url="path/to/video/stream",
    target_video_path="path/to/output/video",
    polygons=[np.array([[x1, y1], [x2, y2], ...])],
    confidence_threshold=0.3,
    iou_threshold=0.7,
    resize=2560
)
```

Replace the paths and parameters with those relevant to your specific setup.

## Infrastructure 📦 

The system's infrastructure is designed to be robust and scalable:

- **Client/Router**: Manages incoming video streams securely.
- **Video Server (1935)**: Processes video streams and feeds data into the analysis pipeline.
- **Prediction Servers**: Perform real-time object detection and behavior analysis.
- **Main Database**: Stores configuration data and analysis results.
- **Message Queuing**: Handles communication between services for asynchronous processing.
- **APIs**: Facilitate user authentication and data operations (Auth API [8080], DOT API [8080]).
- **Frontend Application**: Allows users to interact with the system, view analytics, and manage data.
- **Cloud Datastore**: Provides scalable storage for application data in the cloud.

<p align="center">
  <a href="#">
    <img src="https://github.com/AdonaiVera/human-behavior-analysis/img/example_infraestructure.png" alt="" width="90%">
  </a>
</p>

## Next steps
1. Integrate with yolo-world
2. Annomized the users
3. Tracking people in multiples cameras.
3.1. 

## Contributing 🤓

Interested in contributing? Great! Here's how you can:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Acknowledgments 🎁

If you are interested in building on top of this, feel free to reach out :) 
* **Adonai Vera** - *AI ML Subterra AI* - [AdonaiVera](https://github.com/AdonaiVera)


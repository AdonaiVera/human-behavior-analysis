from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import numpy as np

# Define client-specific variables
camera_entry = [{
    "url": "/Users/adonaivera/Documents/ai-model-dot/input/cam5Parallel_1.mp4",
    "confidence_threshold": 0.3,
    "target_video_path": "/Users/adonaivera/Documents/ai-model-dot/output/cam5Parallel_1.mp4",
    "switch_model": "models/yolov8m.pt",
    "algorithm": [0],
    "iou_threshold": 0.7,
    "resize": 2560,
    "src_points": [(15, 1352), (585, 71), (1003, 46), (2528, 1406)],
    "dst_points": [(44, 8), (608, 12), (609, 465), (47, 465)],
    "polygons": [
        np.array([[0, 1000], [640, 1000], [640, 1436], [0, 1436]], np.int32),
        np.array([[0, 410], [746, 410], [684, 872], [0, 870]], np.int32),
        np.array([[0, 370], [0, 0], [784, 0], [760, 370]], np.int32),
        np.array([[836, 4], [1034, 4], [2456, 1436], [786, 1431]], np.int32),
    ]
},
{
    "url": "/Users/adonaivera/Documents/ai-model-dot/input/cam6Parallel_1.mp4",
    "confidence_threshold": 0.3,
    "target_video_path": "/Users/adonaivera/Documents/ai-model-dot/output/cam6Parallel_1.mp4",
    "switch_model": "models/yolov8m.pt",
    "algorithm": [0],
    "iou_threshold": 0.7,
    "resize": 2560,
    "src_points": [(29, 1402), (859, 201), (1481, 161), (2280, 1414)],
    "dst_points": [(18, 11), (25, 460), (623, 12), (623, 462)],
    "polygons": [
        np.array([[0, 810],[1062,909],[941, 1436],[0,1430]], np.int32),
        np.array([[0, 714],[1076, 822],[1184,376],[436,372]], np.int32),
        np.array([ [482, 326],[862, 90],[1254,90],[1184,320]], np.int32),
    ]
}]

def process_camera_entries_in_parallel(camera_entries: List[Dict]) -> None:
    # Use ProcessPoolExecutor to execute tasks in parallel
    with ProcessPoolExecutor() as executor:
        # Schedule the process_camera_entry function to be executed for each camera entry
        futures = [executor.submit(process_camera_entry, entry) for entry in camera_entries]

        # Wait for all the futures to complete
        for future in as_completed(futures):
            try:
                # Get the result of the future
                future.result()
            except Exception as e:
                print(f"Camera entry processing failed: {e}")


def process_camera_entry(entry: Dict) -> None:
    from app import process_video 
    
    # Extract necessary variables from the entry, add more as needed
    stream_url = entry.get("url")
    confidence_threshold = entry.get("confidence_threshold")
    iou_threshold = entry.get("iou_threshold")
    resize = entry.get("resize")
    polygons = entry.get("polygons")
    target_video_path = entry.get("target_video_path")
    model = entry.get("switch_model")
    src_points = entry.get("src_points")
    dst_points = entry.get("dst_points")

    # Call the model processing function
    process_video(
        source_weights_path=model,
        stream_url=stream_url,
        target_video_path=target_video_path,
        polygons=polygons,
        confidence_threshold=confidence_threshold,
        resize=resize,
        iou_threshold=iou_threshold,
        src_points=src_points,
        dst_points=dst_points
    )

def main():
    # Process the camera entries in parallel
    process_camera_entries_in_parallel(camera_entry)

if __name__ == "__main__":
    main()
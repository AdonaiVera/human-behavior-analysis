from typing import List, Tuple
import cv2
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import os
from methods.save_db import RandomFrameCapturer
from methods.bird_eyes import BirdEyeTransformer
def process_video(
    source_weights_path: str,
    stream_url: str,
    target_video_path: str,
    polygons: List[np.ndarray],
    src_points: List[Tuple[int, int]],
    dst_points: List[Tuple[int, int]],
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    resize: int = 2560,

) -> None:
    
    model = YOLO(source_weights_path)
    colors = sv.ColorPalette.default()

    video_info = sv.VideoInfo.from_video_path(stream_url)

    zones = [
        sv.PolygonZone(
            polygon=polygon, 
            frame_resolution_wh=video_info.resolution_wh
        )
        for polygon
        in polygons
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone, 
            color=colors.by_idx(index), 
            thickness=6,
            text_thickness=8,
            text_scale=4
        )
        for index, zone
        in enumerate(zones)
    ]

    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index), 
            thickness=4, 
            text_thickness=4, 
            text_scale=2
            )
        for index
        in range(len(polygons))
    ]


    # Initialize the byte tracker
    tracker = sv.ByteTrack(show_time=True)

    # Initialize the video frame generator
    frame_generator = sv.get_video_frames_generator(source_path=stream_url, stride=10)

    # Initialize the label annotator
    label_annotator = sv.LabelAnnotator()

    # Initialize the CSV sink
    csv_sink = sv.CSVSink('output/detections_{}.csv'.format(stream_url.split('/')[-1]))
    
    # Initialize the random frame capturer
    frame_capturer = RandomFrameCapturer(interval_minutes=1)
    if not os.path.exists(frame_capturer.output_dir):
        os.makedirs(frame_capturer.output_dir)

    # Initialize the bird-eye transformer
    output_size = (640, 480)  # Desired size of the bird-eye view image
    bird_eye_transformer = BirdEyeTransformer(src_points, dst_points, output_size)


    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink, csv_sink:
        for frame_number, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            #detections = detections[(detections_track.class_id == 0) & (detections_track.confidence > confidence_threshold)]
            
            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
                frame = zone_annotator.annotate(scene=frame)

                frame = label_annotator.annotate(
                    scene=frame, detections=detections
                )

            # Write the frame to the video sink
            sink.write_frame(frame=frame)

            # Append the detections to the CSV file
            csv_sink.append(detections, custom_data={'frame_number': frame_number})

            # Capture the frame if the condition is met
            if frame_capturer.should_capture():
                frame_capturer.capture_frame(frame, frame_number)

            # Transform detections to bird-eye view
            transformed_points = bird_eye_transformer.transform(detections)

            # Generate bird-eye view image
            bird_eye_image = bird_eye_transformer.draw_bird_eye_view(transformed_points)

            # Display the processed frame
            cv2.imshow("Bird's-Eye View {}".format(stream_url.split('/')[-1]), cv2.rotate(bird_eye_image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            cv2.imshow(stream_url, cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
                break

    cv2.destroyAllWindows()
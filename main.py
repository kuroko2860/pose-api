import cv2
import base64
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from threading import Lock
from mmpose.apis import init_model, inference_topdown, inference_pose_lifter_model
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from flask_cors import CORS
from copy import deepcopy

# ---- Configuration ----
MAX_PERSONS_PER_FRAME = 3  # Maximum number of persons to process per frame
MIN_CONFIDENCE_SCORE = 0.6  # Minimum confidence score threshold


def filter_top_persons(pose2d_results, max_persons=MAX_PERSONS_PER_FRAME):
    """Filter to keep only top N persons based on bbox area and confidence score."""
    if not pose2d_results:
        return pose2d_results
    
    all_persons = []
    
    # Extract all persons from all samples
    for sample in pose2d_results:
        inst = sample.pred_instances
        for i in range(len(inst.bboxes)):
            bbox = inst.bboxes[i]
            score = float(inst.scores[i])
            
            # Skip persons with low confidence
            if score < MIN_CONFIDENCE_SCORE:
                continue
            
            # Calculate bbox area
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Skip persons with very small bbox area (likely false positives)
            if bbox_area < 1000:  # Minimum area threshold
                continue
            
            # Create person data with score and area
            person_data = {
                'sample': sample,
                'person_index': i,
                'bbox': bbox,
                'score': score,
                'area': bbox_area,
                'combined_score': score * bbox_area  # Weighted combination
            }
            all_persons.append(person_data)
    
    # Sort by combined score (confidence * area) in descending order
    all_persons.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Take top N persons
    top_persons = all_persons[:max_persons]
    
    print(f"Person filtering: {len(all_persons)} candidates -> {len(top_persons)} selected")
    if top_persons:
        print("Top persons:")
        for i, person in enumerate(top_persons):
            print(f"  {i+1}. Score: {person['score']:.3f}, Area: {person['area']:.0f}, Combined: {person['combined_score']:.0f}")
    
    # Reconstruct pose2d_results with only top persons
    if not top_persons:
        return []
    
    # Group by original sample
    sample_groups = {}
    for person in top_persons:
        sample_id = id(person['sample'])
        if sample_id not in sample_groups:
            sample_groups[sample_id] = {
                'sample': person['sample'],
                'indices': []
            }
        sample_groups[sample_id]['indices'].append(person['person_index'])
    
    # Create filtered samples
    filtered_results = []
    for group in sample_groups.values():
        sample = group['sample']
        indices = group['indices']
        
        # Create new sample with only selected persons
        filtered_sample = deepcopy(sample)
        inst = sample.pred_instances
        
        # Filter instances by indices
        filtered_inst = type(inst)(
            bboxes=inst.bboxes[indices],
            scores=inst.scores[indices],
            keypoints=inst.keypoints[indices]
        )
        filtered_sample.pred_instances = filtered_inst
        filtered_results.append(filtered_sample)
    
    return filtered_results


def split_to_person_samples(pose2d_sample):
    persons = []
    inst = pose2d_sample.pred_instances
    for i in range(len(inst.bboxes)):
        person_sample = deepcopy(pose2d_sample)
        # keep only one person
        person_sample.pred_instances = inst[i : i + 1]
        persons.append(person_sample)
    print("detect person: ", len(persons))
    return persons


def track_pose2d_results(pose2d_results, image, tracker):
    """Attach track IDs to 2D results using SORT/DeepSORT."""

    per_person_samples = []
    detections = []

    for sample in pose2d_results:  # usually len=1 (per image)
        persons = split_to_person_samples(sample)
        per_person_samples.extend(persons)

        inst = sample.pred_instances
        for i in range(len(inst.bboxes)):
            bbox = inst.bboxes[i]
            score = float(inst.scores[i])
            detections.append(([bbox[0], bbox[1], bbox[2], bbox[3]], score, "person"))

    # update tracker
    tracks = tracker.update_tracks(detections, frame=image)

    # assign track IDs back
    tracked_samples = []
    for person_sample, trk in zip(per_person_samples, tracks):
        # if not trk.is_confirmed():
        #     continue
        person_sample.track_id = trk.track_id
        tracked_samples.append(person_sample)

    return tracked_samples


# ---- Init Flask ----
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---- Init Pose Models ----
device = "cuda" if torch.cuda.is_available() else "cpu"
pose2d_model = init_model(
    "/root/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-s_8xb32-600e_body7-640x640.py",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth",
    device=device,
)

pose3d_model = init_model(
    "/root/mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m.py",
    "https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth",
    device=device,
)

# ---- Tracker (global, single client) ----
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    embedder="mobilenet",
    half=True,
)

# ---- GPU lock ----
gpu_lock = Lock()


def run_inference_image(image: np.ndarray):
    """Run full 2D + 3D inference on a single image (OpenCV BGR)."""
    # Step 1: Run 2D pose
    with gpu_lock:
        start_time_real = time.time()
        pose2d_results = inference_topdown(pose2d_model, image)
        end_time_real = time.time()
        print("inference 2d time: ", end_time_real - start_time_real, )

    if len(pose2d_results) == 0:
        return {"2d": [], "3d": [], "analysis": {}}

    # Step 1.5: Filter to top N persons based on bbox area and confidence
    pose2d_results = filter_top_persons(pose2d_results, max_persons=MAX_PERSONS_PER_FRAME)
    
    if len(pose2d_results) == 0:
        return {"2d": [], "3d": [], "analysis": {}}

    # print("pose2d_results", pose2d_results)

    # Step 2: Build detections for tracker
    start_time_real = time.time()
    tracked_results = track_pose2d_results(pose2d_results, image, tracker)
    end_time_real = time.time()
    print("inference 2d+tracker time: ", end_time_real - start_time_real)
    # print("tracked_results", tracked_results)
    # ---- 3D Pose lifting ----
    # print("tracked_results", tracked_results)
    # pose2d_results is [PoseDataSample] (one per image)
    # per_person_samples = []
    # for sample in pose2d_results:
    #     per_person_samples.extend(split_to_person_samples(sample))
    # print("per_person_samples", per_person_samples)
    track_map = {}
    for i, t in enumerate(tracked_results):
        # if not t.is_confirmed():
        #     continue
        track_id = t.track_id
        track_map[i] = track_id

    with gpu_lock:
        start_time_real = time.time()
        pose3d_results = inference_pose_lifter_model(
            pose3d_model, [tracked_results], image_size=image.shape[:2]
        )
        end_time_real = time.time()
        print("inference 3d time: ", end_time_real - start_time_real)

    # print("pose3d_results", pose3d_results)

    # ---- Package results ----
    start_time_real = time.time()
    merged_results = []
    for idx, sample in enumerate(pose3d_results):
        kpts3d = sample.pred_instances.keypoints[0].tolist()
        kpts2d = tracked_results[idx].pred_instances.keypoints.tolist()
        bbox = tracked_results[idx].pred_instances.bboxes[0].tolist()

        track_id = track_map.get(idx, -1)

        # ---- Example analytics: knee angle ----
        # kpts_np = np.array(kpts3d)
        # hip, knee, ankle = kpts_np[11], kpts_np[13], kpts_np[15]  # left leg
        # v1 = hip - knee
        # v2 = ankle - knee
        # angle = float(
        #     np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))
        # )

        merged_results.append(
            {
                "track_id": track_id,
                "keypoints_2d": kpts2d[0],
                "keypoints_3d": kpts3d[0],
                "bbox":bbox,
            }
        )

    end_time_real = time.time()
    print("merge time: ", end_time_real - start_time_real)

    return merged_results


# ---- REST API: upload image ----
@app.route("/image", methods=["POST"])
def handle_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    result = run_inference_image(image)
    
    # Encode the processed image as JPEG binary
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_binary = buffer.tobytes()
    
    # Return both pose results and processed image
    return jsonify({
        "poses": result,
        "image": base64.b64encode(image_binary).decode('utf-8'),  # Base64 for JSON response
        "success": True
    })


# health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return "OK"


# ---- WebSocket Events ----
@socketio.on("frame")
def handle_frame(data):
    # Handle both base64 and binary data
    if isinstance(data, str):
        # Base64 encoded data
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    else:
        # Binary data
        nparr = np.frombuffer(data, np.uint8)
    
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = run_inference_image(image)
    
    # Encode the processed image as JPEG binary
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_binary = buffer.tobytes()
    
    # Send both image and pose results
    emit("pose_result", {
        "image": image_binary,
        "poses": result,
        "success": True
    })


if __name__ == "__main__":
    print("Server started")
    # Use gevent WSGI server with websocket support
    socketio.run(app, host="0.0.0.0", port=5000)

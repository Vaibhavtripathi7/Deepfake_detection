import os
import glob
import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import shutil

RAW_DATASET_BASE_PATH = '/home/vaibhav/PROJECTS/deepfake_dataset'

PROCESSED_DATASET_BASE_PATH = '/home/vaibhav/PROJECTS/processed_dataset_faces_mediapipe'

OUTPUT_FACE_VIDEO_SIZE = (112, 112)
FRAMES_TO_PROCESS_PER_VIDEO = 150
MIN_DETECTION_CONFIDENCE = 0.5

mp_face_detection = mp.solutions.face_detection


def frame_extract(video_path):
    vidObj = cv2.VideoCapture(video_path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image
        else:
            break
    vidObj.release()


def create_face_videos_mediapipe(video_path_list, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"Outputting processed videos to: {output_dir_path}")

    already_present_count = len(glob.glob(os.path.join(output_dir_path, '*.mp4')))
    print(f"Number of videos already processed in this directory: {already_present_count}")

    with mp_face_detection.FaceDetection(min_detection_confidence=MIN_DETECTION_CONFIDENCE) as face_detection:
        for video_path in tqdm(video_path_list,
                               desc=f"Processing Videos in {os.path.basename(os.path.dirname(output_dir_path))}/{os.path.basename(output_dir_path)}"):
            video_filename = os.path.basename(video_path)
            out_video_path = os.path.join(output_dir_path, video_filename)

            if os.path.exists(out_video_path):
                continue

            cap_temp = cv2.VideoCapture(video_path)
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps): fps = 30.0
            cap_temp.release()

            writer = None
            frames_processed_count = 0
            for idx, frame in enumerate(frame_extract(video_path)):
                if FRAMES_TO_PROCESS_PER_VIDEO is not None and idx >= FRAMES_TO_PROCESS_PER_VIDEO:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    xmin = max(0, int(bboxC.xmin * iw))
                    ymin = max(0, int(bboxC.ymin * ih))
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    xmax = min(iw, xmin + w)
                    ymax = min(ih, ymin + h)

                    if xmax > xmin and ymax > ymin:
                        face_image = frame[ymin:ymax, xmin:xmax]
                        try:
                            resized_face = cv2.resize(face_image, OUTPUT_FACE_VIDEO_SIZE)
                            if writer is None:
                                writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'MP4V'), float(fps),
                                                         OUTPUT_FACE_VIDEO_SIZE)
                            writer.write(resized_face)
                            frames_processed_count += 1
                        except Exception as e:
                            print(f"Error processing frame for {video_filename}: {e}")
                            pass

            if writer is not None:
                writer.release()

            if frames_processed_count == 0 and os.path.exists(out_video_path):
                if os.path.getsize(out_video_path) < 1000:
                    print(f"Warning: No faces processed for {video_filename}. Removing empty output: {out_video_path}")
                    os.remove(out_video_path)
                elif frames_processed_count == 0:
                    print(
                        f"Warning: No faces processed for {video_filename}, but output file seems to have data or wasn't created.")


def main():
    raw_train_real_path = os.path.join(RAW_DATASET_BASE_PATH, 'train/real')
    raw_train_fake_path = os.path.join(RAW_DATASET_BASE_PATH, 'train/fake')
    raw_test_real_path = os.path.join(RAW_DATASET_BASE_PATH, 'test/real')
    raw_test_fake_path = os.path.join(RAW_DATASET_BASE_PATH, 'test/fake')

    if not os.path.isdir(raw_train_real_path):  # Basic check
        print(f"ERROR: Raw dataset path not found or incorrect: {RAW_DATASET_BASE_PATH}")
        print("Please ensure RAW_DATASET_BASE_PATH is set correctly and contains train/real, train/fake, etc.")
        return

    train_real_videos = sorted(glob.glob(os.path.join(raw_train_real_path, '*.mp4')))
    train_fake_videos = sorted(glob.glob(os.path.join(raw_train_fake_path, '*.mp4')))
    test_real_videos = sorted(glob.glob(os.path.join(raw_test_real_path, '*.mp4')))
    test_fake_videos = sorted(glob.glob(os.path.join(raw_test_fake_path, '*.mp4')))

    print(f"Found {len(train_real_videos)} raw train real videos.")
    print(f"Found {len(train_fake_videos)} raw train fake videos.")
    print(f"Found {len(test_real_videos)} raw test real videos.")
    print(f"Found {len(test_fake_videos)} raw test fake videos.")

    processed_train_real_output_path = os.path.join(PROCESSED_DATASET_BASE_PATH, 'train/real_faces')
    processed_train_fake_output_path = os.path.join(PROCESSED_DATASET_BASE_PATH, 'train/fake_faces')
    processed_test_real_output_path = os.path.join(PROCESSED_DATASET_BASE_PATH, 'test/real_faces')
    processed_test_fake_output_path = os.path.join(PROCESSED_DATASET_BASE_PATH, 'test/fake_faces')

    if os.path.exists(PROCESSED_DATASET_BASE_PATH):
        pass
    os.makedirs(processed_train_real_output_path, exist_ok=True)
    os.makedirs(processed_train_fake_output_path, exist_ok=True)
    os.makedirs(processed_test_real_output_path, exist_ok=True)
    os.makedirs(processed_test_fake_output_path, exist_ok=True)

    print("\nProcessing Train Real videos...")
    create_face_videos_mediapipe(train_real_videos, processed_train_real_output_path)
    print("\nProcessing Train Fake videos...")
    create_face_videos_mediapipe(train_fake_videos, processed_train_fake_output_path)
    print("\nProcessing Test Real videos...")
    create_face_videos_mediapipe(test_real_videos, processed_test_real_output_path)
    print("\nProcessing Test Fake videos...")
    create_face_videos_mediapipe(test_fake_videos, processed_test_fake_output_path)

    print("\n--- Preprocessing Complete (MediaPipe, Local) ---")
    print(f"Processed videos are saved in: {PROCESSED_DATASET_BASE_PATH}")

    print("--- Sanity Check: Counts of Processed Videos ---")
    processed_train_real_count = len(glob.glob(os.path.join(processed_train_real_output_path, '*.mp4')))
    processed_train_fake_count = len(glob.glob(os.path.join(processed_train_fake_output_path, '*.mp4')))
    processed_test_real_count = len(glob.glob(os.path.join(processed_test_real_output_path, '*.mp4')))
    processed_test_fake_count = len(glob.glob(os.path.join(processed_test_fake_output_path, '*.mp4')))

    print(f"Processed Train Real videos: {processed_train_real_count}")

    print(f"Processed Train Fake videos: {processed_train_fake_count}")
    print(f"Processed Test Real videos: {processed_test_real_count}")
    print(f"Processed Test Fake videos: {processed_test_fake_count}")


if __name__ == '__main__':
    main()
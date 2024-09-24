import os

import cv2
import hydra
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from omegaconf import DictConfig


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

    # draw landmarks in original image
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
    )

    # draw landmarks in white background
    white_background_image = np.ones_like(rgb_image) * 255
    solutions.drawing_utils.draw_landmarks(
        image=white_background_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=white_background_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=white_background_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
    )

    return annotated_image, white_background_image


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    print("Loading FaceLandscapeer...")
    model_path = "models/FaceLandscapeer/face_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if f.endswith(".png") and not f.startswith("grid")
        )
    else:
        generated_image_paths = [cfg.generated_image]

    print(f"Processing {cfg.reference_image}...")
    reference_image = cv2.imread(cfg.reference_image)
    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_image_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=reference_image_rgb
    )

    detection_result = detector.detect(reference_image_mp)

    ref_annotated_image, ref_white_background_image = draw_landmarks_on_image(
        reference_image_rgb, detection_result
    )

    ref_annotated_image = cv2.cvtColor(ref_annotated_image, cv2.COLOR_BGR2RGB)
    ref_white_background_image = cv2.cvtColor(
        ref_white_background_image, cv2.COLOR_BGR2RGB
    )

    cv2.imwrite(f"{cfg.output_dir}/reference_landmarks.png", ref_annotated_image)
    cv2.imwrite(
        f"{cfg.output_dir}/reference_landmarks_white.png", ref_white_background_image
    )

    templete_matrix = detection_result.facial_transformation_matrixes

    result_dict = {}

    for idx, generated_image_path in enumerate(generated_image_paths):
        print(f"Processing {generated_image_path}...")
        generated_image = cv2.imread(generated_image_path)
        basename = os.path.basename(os.path.splitext(generated_image_path)[0])

        if generated_image is None:
            raise ValueError(f"Failed to load image from {generated_image_path}")

        image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = detector.detect(image_mp)

        annotated_image, white_background_image = draw_landmarks_on_image(
            image_rgb, detection_result
        )

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        white_background_image = cv2.cvtColor(white_background_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"{cfg.output_dir}/{basename}_landmarks.png", annotated_image)
        cv2.imwrite(
            f"{cfg.output_dir}/{basename}_landmarks_white.png", white_background_image
        )

        target_matrix = detection_result.facial_transformation_matrixes

        print(f"Comparing {cfg.reference_image} and {generated_image_path}...")
        difference = np.linalg.norm(templete_matrix[0] - target_matrix[0], ord="fro")
        print(f"Difference: {difference:.4f}")

        result_dict[generated_image_path] = difference

    result_df = pd.DataFrame(result_dict.items(), columns=["image_path", "difference"])
    result_df.to_csv(f"{cfg.output_dir}/result.csv", index=False)


if __name__ == "__main__":
    main()

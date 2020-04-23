import eos
import numpy as np
import os
import time


def generate_gt(landmarks):

    landmarks = read_pts(landmarks)
    image_width = 64
    image_height = 64

    model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")

    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())

    landmark_mapper = eos.core.LandmarkMapper('share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('share/sfm_model_contours.json')

    (tx, ty, scale, yaw, roll, pitch, pca_shape_coefficients, expression_coefficients) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)

    # gt_result = {"tx": tx, "ty": ty, "scale": scale, "yaw": yaw, "roll": roll, "pitch": pitch, "pca_shape_coefficients": pca_shape_coefficients, "expression_coefficients": expression_coefficients}
    gt_result = [tx, ty, scale, yaw, roll, pitch, pca_shape_coefficients, expression_coefficients]
    return gt_result

def read_pts(landmarks):
    """A helper function to read the 68 ibug landmarks from numpy array."""
    eos_landmarks = []
    for idx, point in enumerate(landmarks):
        eos_landmarks.append(eos.core.Landmark(str(idx), point))

    return eos_landmarks

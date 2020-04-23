import eos
import numpy as np
import os
import time


def generate_gt(file_path, save_path):
    landmarks_name = os.listdir(file_path)
    if '.DS_Store' in landmarks_name:
        landmarks_name.remove('.DS_Store')
    for landmark_name in landmarks_name:
        landmark_full_path = file_path + '/' + landmark_name

        landmarks = read_pts(landmark_full_path)
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

        gt_result = {"tx": tx, "ty": ty, "scale": scale, "yaw": yaw, "roll": roll, "pitch": pitch, "pca_shape_coefficients": pca_shape_coefficients, "expression_coefficients": expression_coefficients}
        # print(gt_result)
        np.save(save_path + str(landmark_name.split('.')[0]), gt_result)


def read_pts(filename):
    """A helper function to read the 68 ibug landmarks from a .npy file."""
    landmarks = np.load(filename)
    eos_landmarks = []
    for idx, point in enumerate(landmarks):
        eos_landmarks.append(eos.core.Landmark(str(idx), point))

    return eos_landmarks


if __name__ == "__main__":
    save_path = 'ground_truth/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    start = time.time()
    generate_gt(file_path='pts_result', save_path=save_path)
    end = time.time()
    print('Operate Finished | Costed time:{} s'.format(end-start))

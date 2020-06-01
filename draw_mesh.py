import eos
from cv2 import cv2 as cv2
import glm
import numpy as np


modelfile = "share/sfm_shape_3448.bin"
imgfile = "3330.png"
blendshapesfile = "share/expression_blendshapes_3448.bin"
outputbasename = "3330out"

pca_shape_coefficients = {-8.7994e-02, -2.1412e-01,  1.9334e-01, -9.0814e-02,
         1.0191e-01, -1.1960e-01,  1.5348e-01, -2.3314e-03, -4.4998e-02,
         1.1687e-01,  1.1270e-01, -7.0829e-02, -2.8223e-02, -7.1665e-02,
        -1.0576e-01, -5.3201e-02, -1.5072e-01, -1.0871e-01,  6.5697e-02,
         6.7229e-02,  2.9475e-02,  1.3967e-01,  8.0471e-02, -1.1813e-02,
         1.4631e-02,  6.1900e-02,  4.7204e-02,  1.7153e-01, -6.4420e-03,
         2.8472e-02, -3.2023e-02,  6.8777e-02,  1.7580e-01, -6.1445e-02,
        -3.3121e-02,  6.1041e-02,  1.5347e-02, -8.8514e-02,  6.1835e-02,
        -1.4587e-01, -4.7661e-03,  4.8044e-02,  6.8582e-04, -5.4018e-02,
         2.6654e-02, -3.9256e-02,  4.7049e-02,  2.0279e-02,  4.2696e-02,
        -4.9311e-03,  6.1114e-02, -4.3168e-02, -5.0442e-02, -4.8499e-03,
         9.7734e-03, -1.1798e-01, -2.4533e-02, -2.3651e-02,  8.2454e-03,
        -4.1688e-02,  1.4737e-02,  3.9384e-02,  2.2273e-02}

blendshape_coefficients = {-9.0485e-02,
        2.1778e-01,  2.3190e-01, -1.3723e-01,  6.1431e-01, -1.6431e-01}

yaw = 8.2123e+00          # yaw
roll = -4.3990e+00        # roll
pitch = -5.9803e+00       # pitch

tx = 1.2004e+02
ty = 1.2098e+02
scale = 2.7409e-01

image = cv2.imread(imgfile)
outimg = image.copy()

image_width = image.shape[0]
image_height = image.shape[1]


model = eos.morphablemodel.load_model(modelfile);
blendshapes = eos.morphablemodel.load_blendshapes(blendshapesfile);

morphable_model = eos.morphablemodel.MorphableModel(model.get_shape_model(),
                                           blendshapes,
                                           model.get_color_model(),
                                           None,
                                           model.get_texture_coordinates())

rot_mtx_x = glm.rotate(glm.mat4(1.0), pitch/180*3.14159, glm.vec3(1.0, 0.0, 0.0))
rot_mtx_y = glm.rotate(glm.mat4(1.0), yaw/180*3.14159, glm.vec3(0.0, 1.0, 0.0))
rot_mtx_z = glm.rotate(glm.mat4(1.0), roll/180*3.14159, glm.vec3(0.0, 0.0, 1.0))

rotation_matrix = rot_mtx_z * rot_mtx_x * rot_mtx_y
rotation_matrix = glm.mat3x3(rotation_matrix)

current_pose = eos.fitting.ScaledOrthoProjectionParameters(rotation_matrix, tx, ty, scale) #rotation_matrix, tx, ty, scale
# current_pose.R = rotation_matrix
# current_pose.s = scale
# current_pose.tx = tx
# current_pose.ty = ty

rendering_params = eos.fitting.RenderingParameters(current_pose, image_width, image_height)

current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients)

# assert(morphable_model.has_separate_expression_model())

current_combined_shape = current_pca_shape + eos.morphablemodel.draw_sample(morphable_model.get_expression_model().value(), blendshape_coefficients)

current_mesh = eos.morphablemodel.sample_to_mesh(
    current_combined_shape,
    morphable_model.get_color_model().get_mean(),
    morphable_model.get_shape_model().get_triangle_list(),
    morphable_model.get_color_model().get_triangle_list(),
    morphable_model.get_texture_coordinates(),
    morphable_model.get_texture_triangle_indices())

render.draw_wireframe(outimg, current_mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                       eos.fitting.get_opencv_viewport(image_width, image_height))

cv2.imwrite(outputbasename + ".png", outimg)

def are_vertices_ccw_in_screen_space(v0, v1, v2):
    dx01 = v1[0] - v0[0]
    dy01 = v1[1] - v0[1]
    dx02 = v2[0] - v0[0]
    dy02 = v2[1] - v0[1]

    res = dx01 * dy02 - dy01 * dx02 < 0

    return res

def draw_wireframe(image, mesh, modelview, projection, viewport, color = cv2.Scalar(0, 255, 0, 255)):
    for triangle in mesh.tvi:
        p1 = glm.project(
            (mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]),
            modelview, projection, viewport);
        p2 = glm.project(
            (mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]),
            modelview, projection, viewport);
        p3 = glm.project(
            (mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]),
            modelview, projection, viewport);
        if are_vertices_ccw_in_screen_space(glm.vec2(p1), glm.vec2(p2), glm.vec2(p3)):
            cv2.line(image, cv2.Point(p1.x, p1.y), cv2.Point(p2.x, p2.y), color)
            cv2.line(image, cv2.Point(p2.x, p2.y), cv2.Point(p3.x, p3.y), color)
            cv2.line(image, cv2.Point(p3.x, p3.y), cv2.Point(p1.x, p1.y), color)

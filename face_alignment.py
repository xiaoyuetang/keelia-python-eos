# -*-coding: utf-8-*-
import cv2
import numpy as np
import math
from collections import defaultdict
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# from KeypointsDataset import KeypointsDataset
# from KeypointsDataset import Rescale

# face_dataset = KeypointsDataset(csv_file='ytb/training_frames_keypoints.csv',
#                                 root_dir='ytb/training/')
#
# data = face_dataset[8]
#
# img_name = data['image_name']
# image_array = cv2.imread(img_name)
#
# key_pts = data['keypoints']

def generate_lmks(key_pts):
    landmarks = []
    for pts in key_pts:
        landmarks.append(tuple(pts))
    return landmarks


def tuple2array(key_pts):
    landmarks = []
    for pts in key_pts:
        landmarks.append(list(pts))
    return np.array(landmarks)


def generate_dict(landmarks):
    landmarks_dict = {'left_eye':[], 'right_eye':[], 'chin':[], 'left_eyebrow':[], 'right_eyebrow':[], 'nose_tip':[], 'nose_bridge':[], 'top_lip':[], 'bottom_lip':[]}

    for i in range(36, 42): #6x2=12
        landmarks_dict['left_eye'].append(landmarks[i])
        landmarks_dict['right_eye'].append(landmarks[i+6])

    for i in range(17): #17
        landmarks_dict['chin'].append(landmarks[i])

    for i in range(17, 22): #5x3=15
        landmarks_dict['left_eyebrow'].append(landmarks[i])
        landmarks_dict['right_eyebrow'].append(landmarks[i+5])
        landmarks_dict['nose_tip'].append(landmarks[i+14])

    for i in range(27, 31): #4
        landmarks_dict['nose_bridge'].append(landmarks[i])

    top_lip_pts = [49,50,51,52,53,54,55,61,62,63,64,65]
    bottom_lip_pts = [49,61,68,67,66,65,55,56,57,58,59,60]

    for i in top_lip_pts:
        landmarks_dict['top_lip'].append(landmarks[i-1])

    for i in bottom_lip_pts:
        landmarks_dict['bottom_lip'].append(landmarks[i-1])

    return landmarks_dict

def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    draw.point(landmarks)
    imshow(origin_img)

# visualize_landmark(image_array=image_array, landmarks=landmarks)
# plt.show()

def align_face(image_array, landmarks_dict):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks_dict['left_eye']
    right_eye = landmarks_dict['right_eye']

    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]

    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi

    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)

    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

# aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks_dict=generate_dict(landmarks))

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = []
    for landmark in landmarks:
        rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
        rotated_landmarks.append(rotated_landmark)
    return rotated_landmarks

# rotated_landmarks = rotate_landmarks(landmarks=landmarks,
#                                          eye_center=eye_center, angle=angle, row=image_array.shape[0])
# visualize_landmark(image_array=aligned_face,landmarks=rotated_landmarks)
# plt.show()

def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']), np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")

    lip_landmark = np.concatenate([np.array(landmarks['top_lip']), np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")

    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

# cropped_face, left, top = corp_face(image_array=aligned_face, landmarks=generate_dict(rotated_landmarks))

def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = []
    for landmark in landmarks:
        transferred_landmark = (landmark[0] - left, landmark[1] - top)
        transferred_landmarks.append(transferred_landmark)
    return transferred_landmarks

# transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
# visualize_landmark(image_array=cropped_face,landmarks=transferred_landmarks)
# plt.show()

def face_alignment(image_array, keypoints):
    landmarks = generate_lmks(keypoints)
    aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks_dict=generate_dict(landmarks))
    rotated_landmarks = rotate_landmarks(landmarks=landmarks,
                                             eye_center=eye_center, angle=angle, row=image_array.shape[0])
    cropped_face, left, top = corp_face(image_array=aligned_face, landmarks=generate_dict(rotated_landmarks))
    transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
    transferred_landmarks = tuple2array(transferred_landmarks)

    return cropped_face, transferred_landmarks

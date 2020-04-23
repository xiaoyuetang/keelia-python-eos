# -*-coding: utf-8-*-

import os
import sys
import dlib
import time
import numpy as np
from cv2 import cv2 as cv2


def load_model():
    model_path = 'share/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    landmark = dlib.shape_predictor(model_path)

    return detector, landmark


def generate_landmarks(file_path, save_path):
    imgs = os.listdir(file_path)
    if '.DS_Store' in imgs:
        imgs.remove('.DS_Store')

    for img_name in imgs:
        img_full_path = file_path + '/' + img_name
        img = cv2.imread(img_full_path)

        # 取灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 人脸数rects
        rects = detector(img_gray, 0)
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in landmark(img,rects[i]).parts()])

        np.save(save_path + str(img_name.split('.')[0]), landmarks)

        # for idx, point in enumerate(landmarks):
        #     # 68点的坐标
        #     pos = (point[0, 0], point[0, 1])
        #     print(pos)
        #
        #     # 利用cv2.circle给每个特征点画一个圈，共68个
        #     cv2.circle(img, pos, 5, color=(0, 255, 0))
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)


if __name__ == '__main__':

    detector, landmark = load_model()
    save_path = 'pts_result/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    start = time.time()
    generate_landmarks(file_path='faces_result', save_path=save_path)
    end = time.time()
    print('Operate Finished | Costed time:{} s'.format(end-start))

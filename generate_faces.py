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


def face_aligniment(file_path, save_path):
    '''
    对文件夹中的照片进行批量的人脸对齐操作
    '''
    imgs = os.listdir(file_path)
    if '.DS_Store' in imgs:
        imgs.remove('.DS_Store')
    for img in imgs:
        img_full_path = file_path + '/' + img
        bgr_imgs = cv2.imread(img_full_path)
        if bgr_imgs is None:
            print('Load pics failed !!! Please check file path !!!')
            exit()

        # 照片颜色通道转换：dlib检测的图片空间是RGB， cv2的颜色空间是BGR
        rgb_imgs = cv2.cvtColor(bgr_imgs, cv2.COLOR_BGR2RGB)
        # img_gray = cv2.cvtColor(bgr_imgs, cv2.COLOR_RGB2GRAY)
        face_locations = detector(rgb_imgs, 2)
        if len(face_locations) == 0:
            print('No face detected in pic: {}'.format(img_full_path))
            continue
        else:
            print(len(face_locations))
            # 人脸关键点检测
            face_keypoints = dlib.full_object_detections()
            for location in face_locations:
                face_keypoints.append(landmark(rgb_imgs, location))

            # 人脸对齐
            alignmented_face = dlib.get_face_chips(rgb_imgs, face_keypoints, size=240)

            # 保存对齐后的人脸照片
            for idx, image in enumerate(alignmented_face):
                rgb_img = np.array(image).astype(np.uint8)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path + str(img.split('.')[0]) + str(idx) + '.jpg', bgr_img)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector, landmark = load_model()
    save_path = 'faces_result/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    start = time.time()
    face_aligniment(file_path='ibug', save_path=save_path)
    end = time.time()
    print('Operate Finished | Costed time:{} s'.format(end-start))

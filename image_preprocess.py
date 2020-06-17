import os
import cv2
import dlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
detector = dlib.get_frontal_face_detector()



def face_alignment(face):
    # 预测关键点
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
    shape = predictor(np.uint8(face), rec)
    # left eye, right eye, nose, left mouth, right mouth
    order = [36, 45, 30, 48, 54]
    for j in order:
        x = shape.part(j).x
        y = shape.part(j).y
#        # 计算两眼的中心坐标
    eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
    dx = (shape.part(45).x - shape.part(36).x)
    dy = (shape.part(45).y - shape.part(36).y)
    # 计算角度
    angle = math.atan2(dy, dx) * 180. / math.pi
    # 计算仿射矩阵
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    # 进行仿射变换，即旋转
    RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))

    return RotImg


def crop_faces(plot_images=False, max_images_to_plot=5):
    bad_crop_count = 0
    if not os.path.exists('normalized_images'):
        os.makedirs('normalized_images')
    print ('Cropping faces and saving to %s' % 'normalized_images')
    good_cropped_images = []
    good_cropped_img_file_names = []
    detected_cropped_images = []
    original_images_detected = []
    for file_name in sorted(os.listdir('images')):
        np_img = cv2.imread(os.path.join('images',file_name))
        detected = detector(np_img, 1)
        img_h, img_w, _ = np.shape(np_img)
        original_images_detected.append(np_img)

        if len(detected) != 1:
            bad_crop_count += 1
            continue

        d = detected[0]
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = int(x1 - 0.1 * w)
        yw1 = int(y1 - 0.1 * h)
        xw2 = int(x2 + 0.1 * w)
        yw2 = int(y2 + 0.1 * h)
        cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2)
        
        
        
        cropped_img = face_alignment(cropped_img)
        
        norm_file_path = '%s/%s' % ('normalized_images', file_name)
        cv2.imwrite(norm_file_path, cropped_img)

        good_cropped_img_file_names.append(file_name)

    # save info of good cropped images
    data = pd.read_csv('data.csv')
    filter = data['name'].isin(good_cropped_img_file_names)
    data = data[filter]
    data.to_csv('normalized_data.csv',index=False)
    

    print ('Cropped %d images and saved in %s - info in %s' % (len(original_images_detected), 'normalized_images', 'normalized_data.csv'))
    print ('Error detecting face in %d images - info in Data/unnormalized.txt' % bad_crop_count)

    if plot_images:
        print ('Plotting images...')
        img_index = 0
        plot_index = 1
        plot_n_cols = 3
        plot_n_rows = len(original_images_detected) if len(original_images_detected) < max_images_to_plot else max_images_to_plot
        for row in range(plot_n_rows):
            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(original_images_detected[img_index].astype('uint8'))
            plot_index += 1

            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(detected_cropped_images[img_index])
            plot_index += 1

            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(good_cropped_images[img_index])
            plot_index += 1

            img_index += 1
    plt.show()
    return good_cropped_images



# image cropping method taken from:
# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.CONSTANT,(0,0,0))
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

if __name__ == '__main__':
    crop_faces()

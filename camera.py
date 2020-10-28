import cv2
import dlib
import os
import dlib
from PIL import Image
from skimage import io
from scipy.spatial import distance


    


if __name__ == '__main__':
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    name = "Шалькин Д. О"
    os.chdir(name)
    descript = []
    for i in os.listdir():
        img = io.imread(i)
        dets = detector(img,1)
    for k,d in enumerate(dets):
        shape = sp(img,d)
    discriptor = facerec.compute_face_descriptor(img,shape)
    descript.append(discriptor)
    os.chdir("..")
    while True:
        ret, img = cap.read()
        cv2.imshow("camera", img)
        dets = detector(img,1)
    #находим лицо на фотографии
        for k,d in enumerate(dets):
        # print("Detection {}: Left:{} Top : {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img,d)

        face = facerec.compute_face_descriptor(img,shape) #находим дискриптор лица
        identif = []
        for i in range(len(descript)):
            short = distance.euclidean(face,descript[i])
            if short < 0.6:
                identif.append(1)
            else:
                identif.append(0)
        if sum(identif)/len(identif)> 0.7:
            print("Проходите")
        else:
            print("Внимание!")
        if cv2.waitKey(10) == 27: # Клавиша Esc
            break
    cap.release()
    cv2.destroyAllWindows()
import cv2
import pytesseract as ts
from imutils import contours
import numpy as np



def colly(img,):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    blure = cv2.GaussianBlur(img, (5,5),0)
    return cv2.Canny(blure, 30, 30*3)

def mask(image):
    height = image.shape[0]
    widht = image.shape[1]
    polygon = np.array([(widht//2,0),(widht,0),(widht,height),(widht//2,height)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,np.array([polygon], dtype=np.int64), 1024)
    mask_image = cv2.bitwise_and(image,mask)
    return mask_image


def frames(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,frame_1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    frame_2 =cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    return frame_1, frame_2
def nothing(*arg):
    pass

if __name__ == '__main__':
        
    ts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 24)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    while True:
        _, img = cap.read() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        frame = colly(img)
        frame = mask(frame)

        config= r"--psm 8 --oem 3" #psm 1 7 8

        frame_1,frame_2 = frames(gray)

        combo = cv2.addWeighted(frame_1,1,frame_2,0.5,1)
        cntr, hierarchy  = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntr, _ = contours.sort_contours(cntr)
        frame_text=[]
        for c in cntr:
            area = cv2.contourArea(c)
            if area > 3500:
                x, y, w, h = cv2.boundingRect(c)
                frame_text.append(frame_1[y:y+h , x:x+w])
                frame_text.append(frame_2[y:y+h , x:x+w])
                frame_text.append(combo[y:y+h , x:x+w])
                for i in range(3):
                    txt = ts.image_to_string(frame_text[i],lang="rus", config=config )
                    print(txt.lower())
        #cv2.drawContours(img, cntr, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
        cv2.imshow("Image",img)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) == 27 or cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

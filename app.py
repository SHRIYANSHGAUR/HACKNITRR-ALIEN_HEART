
from flask import Flask, render_template, Response, request
import numpy as np
import cv2
import mediapipe as mp
import time
import os
import hands as htm


app = Flask(__name__)



cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

##CANVAS
imgCanvas=np.zeros((720,1280, 3),np.uint8)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/application')
def application():


    drawColor=(255,0,255)
    xp=0
    yp=0

    detector= htm.handDetector(detectionCon=0.8)
    #read the img files
    folder= "header"
    mylist= os.listdir(folder)
    #print(mylist)

    overlaylist=[]

    for imgpath in mylist:
        image=cv2.imread(f'{folder}/{imgpath}')
        overlaylist.append(image)
    #print(len(overlaylist))
    header= overlaylist[0]
    #print(header)


    while True:
        success, img=cap.read()
        img=cv2.flip(img,1)
    #import all images

    #finding draw_landmarks

        img= detector.findHands(img)
        lmList=detector.findPosition(img, draw=False)
        if len(lmList)!=0:
            #print(lmList)
        #tip of index and middle
        #the 2nd and 3rd val are x ,y corrdinate
        #select from 1:till end
            x1,y1= lmList[8][1:]
            x2,y2= lmList[12][1:]


            fingers=detector.fingerUp()
            #print(fingers)

            if fingers[1] and fingers[2]:
                #xp,yp=0,0
            ###cv2.rectangle(img, (x1,y1-25), (x2,y2+25), (255,0,255),cv2.FILLED)
                if y1<125:
                    if 250<x1<450:
                        header=overlaylist[0]
                        drawColor=(0,0,255)
                    if 500<x1<750:
                        header=overlaylist[1]
                        drawColor=(255,0,0)
                    if 750<x1<950:
                        header=overlaylist[2]
                        drawColor=(0,255,255)
                    if 1050<x1<1250:
                        header=overlaylist[3]
                        drawColor=(0,0,0)


                cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor,cv2.FILLED)
                #print("SELECTION")
            elif fingers[1]:
                cv2.circle(img, (x1,y1), 15, drawColor,cv2.FILLED)
                if xp==0 and yp==0:
                    xp,yp= x1,y1
###this makes line by  asssigning previous pointer to next pointer
                if drawColor==(0,0,0):
                    cv2.line(img,(xp,yp), (x1,y1), drawColor, 55)
                    cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor, 55)



                cv2.line(img,(xp,yp), (x1,y1), drawColor, 15)
                cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor, 15)

                xp,yp= x1,y1
                #print("DWaRA")
##this assigns the pointer back to prevoius location else it forms a radically outward lines from previpus pomiter


    #draw with fingers up
    # selection with 2fingers
    # draw with index
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)

    #sett header
        img[:125, :1280] = header
    #img=cv2.addWeighted(img, 0.5, imgCanvas,0.5,0)
        cv2.imshow("image",img)
    #cv2.imshow("Canvas",imgCanvas)

    #cv2.imshow("Inv",imgInv)

        cv2.waitKey(1)




@app.route('/smile')
def smile():
    pTime = 0
    NUM_FACE = 2


    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            lmlist=[]
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    cx,cy = int(lm.x*iw), int(lm.y*ih)
                    lmlist.append([id,cx,cy])
                smiles=[]
                if len(lmlist)!=0:
                    x1,y1= lmlist[61][1:]

                    x2,y2= lmlist[291][1:]

                    x4,y4= lmlist[317][1:]
                    if y1<y4 and y2<y4:
                        smiles.append(1)
                    else:
                        smiles.append(0)
                    total= smiles.count(1)
                    if total >=1:
                        cv2.putText(img, "  Keep Smiling", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 5)
                    else:
                        cv2.putText(img, " Dont be sad", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 5)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Smile", img)
        cv2.waitKey(1)






if __name__ == '__main__':
    app.run()

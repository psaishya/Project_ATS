from django.shortcuts import render,redirect
import cv2
import numpy as np
from django.http import HttpResponse,StreamingHttpResponse,HttpResponseRedirect
from django.views.decorators import gzip
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db,storage
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from django.template import Context, Template, loader
from django.urls import reverse
from threading import Event

cred = credentials.Certificate("./static/tateserviceaccountkey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://tate-project-default-rtdb.firebaseio.com/',
    'storageBucket':"tate-project.appspot.com"
})
ref=db.reference('Students')


def facedata_capture(request):
    cap=cv2.VideoCapture(0)
    frontal_face_cascade=cv2.CascadeClassifier("./static/haarcascade_frontalface_alt.xml")
    side_face_cascade = cv2.CascadeClassifier("./static/haarcascade_profileface.xml")

    skip=0
    face_data=[]
    face_dataset_path="./face_dataset/"

    if request.method == 'POST':
        name = request.POST.get("name")
        rollno = request.POST.get("rollno")
        batch = request.POST.get("batch")
        program = request.POST.get("program")
    # name=input("Enter the name of student (firstname_lastname) : ")
    # rollno=input("Enter the roll number : ")
    # batch=input("Enter the batch (eg. 2020) : ")
    # program=input("Enter the program (eg. CE,CS) : ")

        file_name=program +'_'+ batch +'_'+ name +'_'+ rollno

        print(file_name)

        while (cap.isOpened()):
            ret,frame=cap.read()
            grey_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            if not ret or frame is None:
                continue

            # if ret==False:
            #     continue
            
            faces_frontal = frontal_face_cascade.detectMultiScale(grey_frame,1.3,5)
            faces_side = side_face_cascade.detectMultiScale(grey_frame, 1.3, 5)

            # faces = list(faces_frontal) 

            faces = list(faces_frontal) + list(faces_side)

            if len(faces)==0:
                continue
            
            k=1
            faces=sorted(faces,key=lambda x: x[2]*x[3],reverse=True)
            
            skip+=1
            for face in faces[:1]:
                x,y,w,h=face
                offset=5
                face_offset=frame[y-offset:y+h+offset,x-offset:x+w+offset]

                face_selection=cv2.resize(face_offset,(100,100))
                
                if skip%5==0:
                    face_data.append(face_selection)
                    print (len(face_data))
                    
                
                # cv2.imshow(str(k),face_selection)
                
                k+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            # cv2.imshow("faces",frame)    
            
            key_pressed=cv2.waitKey(1)
            if key_pressed ==ord('q') or len(face_data)==15 :
                break

        face_data=np.array(face_data)
        print(face_data.shape)
        face_data=face_data.reshape(face_data.shape[0],-1)

        print(face_data.shape)

        np.save(face_dataset_path+file_name,face_data)
        print(f"Dataset saved at : {face_dataset_path+file_name+'.npy'}")

        ################################
        data={}
        data[file_name]={
            'name':name,
            'rollno' :rollno,
            'faculty':program,
            'batch':batch,
            'total_attendance':0,
            'last_attendance_time':"2000-01-01 00:00:00"
            }
        for key,value in data.items():
            ref.child(key).set(value)
            
        bucket=storage.bucket()
        blob=bucket.blob(face_dataset_path+file_name+'.npy')
        blob.upload_from_filename(face_dataset_path+file_name+'.npy')
        ###############################

        cap.release()
        cv2.destroyAllWindows()
        message="Successfully recorded"

        return render(request,"facedata_capture.html",context={'message':message})
    else:
        return render(request, "facedata_capture.html")

    return HttpResponse("This is a default response")

frontal_face_cascade=cv2.CascadeClassifier("./static/haarcascade_frontalface_alt.xml")
side_face_cascade = cv2.CascadeClassifier("./static/haarcascade_profileface.xml")

# Global variable to store the latest frame
latest_frame = None
lock = threading.Lock()

# Function to perform face detection
def detect_faces(frame):
    # Face detection
    faces_frontal = frontal_face_cascade.detectMultiScale(frame, 1.3, 5)
    faces_side = side_face_cascade.detectMultiScale(frame, 1.3, 5)

    # Combine the detected faces from both classifiers
    faces = list(faces_frontal) + list(faces_side)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

# Function to capture video frames
def capture_frames():
    global latest_frame

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not capture video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            latest_frame = detect_faces(frame)

    cap.release()

# Start the video capture thread
video_thread = threading.Thread(target=capture_frames)
video_thread.daemon = True
video_thread.start()
@gzip.gzip_page
def webcam_stream(request):
    global latest_frame

    def generate_frames():
        while True:
            with lock:
                frame = latest_frame

            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def train_model(request):
    face_dataset_path="./face_dataset/"
    face_data=[]
    labels=[] 
    face_id=0
    names={}
    EPOCH=200
    print(type(os.listdir(face_dataset_path)))
    print(len((os.listdir(face_dataset_path))))
    if(len((os.listdir(face_dataset_path))))==0:
        
        return HttpResponse("No data to train")
    for filename in os.listdir(face_dataset_path):
        
        if filename.endswith('.npy'):
            names[face_id]=filename[:-4]
            
            face_data_item=np.load(face_dataset_path+filename)
            print(face_data_item.shape)
            face_data.append(face_data_item)
            
            label=face_id*np.ones((face_data_item.shape[0],))#label for all photos of each person 
            print(label)
            labels.append(label)
            
            face_id +=1
    print(names)
    print(face_data)
    face_dataset=np.concatenate(face_data,axis=0) 
    print(face_dataset)
    #face data will be stacked vertically  
    # The result of this line of code, face_dataset, will be a single NumPy array 
    # that contains the data from all the individual arrays in face_data stacked on top of each other. 
    # The data from each array in face_data is combined into rows of the face_dataset array.
    face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
    print(face_labels)
    print(face_dataset.shape)
    print(face_labels.shape)

    no_of_person=face_id
    print(no_of_person)
    x_train, x_test, y_train, y_test = train_test_split(face_dataset, face_labels, test_size=0.2)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True)
    lr_sch = ReduceLROnPlateau(monitor = 'val_loss', patience= 8, factor = 0.1, verbose = 1, min_lr = 5e-10)

    print(x_test.size)
    print(x_train.size)
    print(x_test.shape)
    print(x_train.shape)
    x_train = x_train.reshape((x_train.shape[0], 100, 100,3))
    x_test = x_test.reshape((x_test.shape[0], 100, 100,3))
    print(x_test.shape)
    print(x_train.shape)

    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    y_train = to_categorical(y_train, num_classes=no_of_person)
    y_test = to_categorical(y_test, num_classes=no_of_person)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print("Shape of input data:", x_train.shape[0])


    model=Sequential(
    [  
        
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        
        BatchNormalization(input_shape=(100, 100, 3)),
        Conv2D(16, (3, 3), activation="relu",input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        # Dropout(0.25),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        # Dropout(0.25),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(units=32,activation="relu"),
        BatchNormalization(),
        
        Dense(units=64,activation="relu"),
        BatchNormalization(),
            
        Dense(units=no_of_person,activation="softmax")
    ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit( x_train,y_train,
            epochs=EPOCH,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, lr_sch],
            shuffle=False)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    savelocation="./static/"
    model.save(savelocation+"facetrainingmodel.keras")
    return render(request, "trained.html")


def training_model(request):
    return render(request, "training_model.html")
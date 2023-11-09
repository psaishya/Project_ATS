from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse,JsonResponse
import cv2
from django.views.decorators import gzip
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from datetime import timedelta
import time
import threading
from threading import Event
from facedata_collection.views import *
import json
from django.core.cache import cache



import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

firebase_initialized = False

message_to_send=""

def initialize_firebase_app():
    global firebase_initialized

    if not firebase_initialized:
        try:
            cred = credentials.Certificate("./static/tateserviceaccountkey.json")
            # Check if the app has already been initialized by checking the length of the apps list
            if len(firebase_admin._apps) == 0:
                firebase_admin.initialize_app(cred,{
                    'databaseURL':'https://tate-project-default-rtdb.firebaseio.com/',
                    'storageBucket':"tate-project.appspot.com"
                })
                firebase_initialized = True
                print("Firebase app initialized successfully")
            else:
                print("Firebase app is already initialized")

        except FileNotFoundError:
            print("Service account key file not found")



# @gzip.gzip_page
# def webcam_streamcheck(request):
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Could not capture video")

#     def generate_frames():
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# Create your views here.

def take_attendance(request):
    print(message_to_send)
    return render(request,"take_attendance.html",context={"message":message_to_send})
    # return HttpResponse(f"<p>{message}</p>")


frontal_face_cascade=cv2.CascadeClassifier("./static/haarcascade_frontalface_alt.xml")
side_face_cascade = cv2.CascadeClassifier("./static/haarcascade_profileface.xml")
model = load_model('./static/facetrainingmodel.keras')
face_dataset_path="./face_dataset/"

names={}
def find_names():
    face_id=0
    for filename in os.listdir(face_dataset_path):
        if filename.endswith('.npy'):
            names[face_id]=filename[:-4]
            face_id+=1
    print(names)
    return names
names=find_names()

# def attendance(request):
#     cap=cv2.VideoCapture(0)
#     # face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
   
#     countframe=0
#     studentid=None
   
#     start_time = None
#     detected_face_name = None

#     while (cap.isOpened()):
#         ret,frame=cap.read()
#         if ret==False:
#             continue
#         grey_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces_frontal = frontal_face_cascade.detectMultiScale(grey_frame,1.3,5)
#         faces_side = side_face_cascade.detectMultiScale(grey_frame, 1.3, 5)
#         faces = list(faces_frontal) + list(faces_side)
#         if(len(faces)==0):
#             start_time = None
#             detected_face_name = None
#             counter=0
     
#         for face in faces:
#             x,y,w,h=face
#             offset=5
#             face_offset=frame[y-offset:y+h+offset,x-offset:x+w+offset]
#             face_selection=cv2.resize(face_offset,(100,100))
#             face_selection = np.expand_dims(face_selection, axis=0)
#             face_selection = np.expand_dims(face_selection, axis=-1)  

#             result=model.predict(face_selection)

#             score=tf.nn.softmax(result[0])
#             out=np.argmax(score)
#             print(out)
#             if start_time is None or (names[(out)]!=detected_face_name and len(faces) == 1):
#                 start_time=time.time()
#                 detected_face_name=names[(out)]
             
#             else:
#                 elapsed_time=time.time()-start_time
#                 print(elapsed_time)
#                 if elapsed_time>=2:
#                     print(f"The face detected continuously is {detected_face_name}")
#                     studentid=detected_face_name
                    
#                     if countframe==0:
#                         countframe=1
                  
#                     start_time=None
            
#             text=names[(out)] 
#             cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
#         ####downloading info
#         print(f"countframe============{countframe}")
#         if countframe!=0:
#             if countframe==1:
#                 print(studentid)
                
#                 studentInfo_ref=db.reference(f'Students/{studentid}')
#                 print(studentInfo_ref)
#                 studentInfo = studentInfo_ref.get() 
            
#             if studentInfo is not None:
#                     print(studentInfo)
#                     #update the no. of atendance and last attendance time
#                     timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#                     ref=db.reference(f'Students/{studentid}')
#                     last_attendance_time=time.strptime(studentInfo['last_attendance_time'], '%Y-%m-%d %H:%M:%S')
#                     if last_attendance_time is not None:
#                         current_time = time.time()
#                         last_Time=time.mktime(last_attendance_time)
#                         if (current_time-last_Time)/60 >1 :
#                             print("1 minute ago, attendance possible")
#                             timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#                             studentInfo['total_attendance']+=1
#                             studentInfo['last_attendance_time']=timestamp
#                             ref.child('total_attendance').set(studentInfo['total_attendance'])
#                             ref.child('last_attendance_time').set(studentInfo['last_attendance_time'])
#                             date=time.strftime('%Y-%m-%d')
#                             attendance_ref = db.reference(f'Attendances/{date}/{studentid}')
#                             attendance_ref.set(True)
#                             message="Attendance recorded for "+studentInfo['name']
#                         else:
#                             print("attendance not possible")
#                             countframe=0
#                             message="Attendance not possible now"
#             else:
#                     print(f"No data found for student with ID {studentid}")
#                     message="No data found for student with ID"+studentid
#             countframe +=1
#         cv2.imshow("faces",frame)    
#         key_pressed=cv2.waitKey(1)
#         if key_pressed ==ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     # return render(request,"attendance.html",context={"message":message})  
#     return HttpResponse(f"<p>{message}</p>")


def find_names():
    face_id = 0
    for filename in os.listdir(face_dataset_path):
        if filename.endswith('.npy'):
            names[face_id] = filename[:-4]
            face_id += 1
    print(names)
    return names

# Load the face recognition model and initialize Firebase on server startup
initialize_firebase_app()

req=None
@gzip.gzip_page
def webcam_streamcheck(request):
  
    cap = cv2.VideoCapture(0)
    message_to_send=""

    if not cap.isOpened():
        print("Could not capture video")

    def generate_frames(message_to_send):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with lock:
                frame_with_faces,message_to_send= recognize_faces(frame.copy(),cache.get('message_to_send', ''))

            _, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame_with_faces = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_with_faces + b'\r\n\r\n')
            # print(message_to_send)
            # get_message(message_to_send)
            cache.set('message_to_send', message_to_send)

    # get_message(message_to_send)
    print(message_to_send)
    return StreamingHttpResponse(generate_frames(message_to_send), content_type='multipart/x-mixed-replace; boundary=frame')
       



names={}
names=find_names()


def recognize_faces(frame,message_to_send):
    names=find_names()

    countframe=0
    studentid=""
   
    start_time = None
    detected_face_name = None
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frontal = frontal_face_cascade.detectMultiScale(grey_frame, 1.3, 5)
    faces_side = side_face_cascade.detectMultiScale(grey_frame, 1.3, 5)
    faces = list(faces_frontal) + list(faces_side)
    

    for face in faces:
        x, y, w, h = face
        offset = 5
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_selection = cv2.resize(face_offset, (100, 100))
        face_selection = np.expand_dims(face_selection, axis=0)
        face_selection = np.expand_dims(face_selection, axis=-1)

        result = model.predict(face_selection)
        score = tf.nn.softmax(result[0])
        out = np.argmax(score)

        text = names[out]
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        start_time=time.time()
        detected_face_name=names[(out)]
        elapsed_time=time.time()-start_time
        print(elapsed_time)
        if elapsed_time>=2:
            print(f"The face detected continuously is {detected_face_name}")
        studentid=detected_face_name
        studentInfo_ref=db.reference(f'Students/{studentid}')
                # print(studentInfo_ref)
        studentInfo = studentInfo_ref.get() 
        if studentInfo is not None:
                    print(studentInfo)
                    #update the no. of atendance and last attendance time
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    ref=db.reference(f'Students/{studentid}')
                    last_attendance_time=time.strptime(studentInfo['last_attendance_time'], '%Y-%m-%d %H:%M:%S')
                    if last_attendance_time is not None:
                        current_time = time.time()
                        last_Time=time.mktime(last_attendance_time)
                        if (current_time-last_Time)/60 >1 :
                            print("1 minute ago, attendance possible")
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            studentInfo['total_attendance']+=1
                            studentInfo['last_attendance_time']=timestamp
                            ref.child('total_attendance').set(studentInfo['total_attendance'])
                            ref.child('last_attendance_time').set(studentInfo['last_attendance_time'])
                            date=time.strftime('%Y-%m-%d')
                            attendance_ref = db.reference(f'Attendances/{date}/{studentid}')
                            attendance_ref.set(True)
                            # message="Attendance recorded for "+studentInfo['name']
                            message="Attendance recorded"

                        else:
                            print("attendance not possible")
                            countframe=0
                            message="Attendance not possible now"
        else:
                    print(f"No data found for student with ID {studentid}")
                    message="No data found for student with ID"
                    
        message_to_send=message
        print(message_to_send)
    return frame,message_to_send

def get_message(request):
    with lock:
        message_to_send = cache.get('message_to_send', '')
        print(message_to_send)
    print("function called")
    print(message_to_send)
    print(type(message_to_send))
    message_to_send = message_to_send.strip()
    return JsonResponse({'data': message_to_send})



def show_attended_student(request):
    date=time.strftime('%Y-%m-%d')
    dict_of_Attended={}
    todayattended_ref=db.reference(f'Attendances/{date}')
                # print(studentInfo_ref)
    todayattended = todayattended_ref.get()
    if todayattended is not None:
        print(todayattended)
        ids=list(todayattended.keys())
        sn=1
        print(ids)
        for id in ids:
            splited=id.split('_')
            name=splited[2]
            batch=splited[1]
            rollno=splited[3]
            program=splited[0]
            dict_of_Attended[sn]=name
            sn+=1
        print(dict_of_Attended)
    return JsonResponse({'data': dict_of_Attended})




def view_attendance(request):
    return render(request,"view_attendance.html")


def showattendancebydate(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_date = data.get('selected_date')
            print("Selected Date (from JSON):", selected_date)
            
            response_data = {'result': 'success'}
            dict_of_Attended={}
            attended_ref=db.reference(f'Attendances/{selected_date}')
                # print(studentInfo_ref)
            attended = attended_ref.get()
            if attended is not None:
                print(attended)
                ids=list(attended.keys())
                sn=1
                print(ids)
                for id in ids:
                    splited=id.split('_')
                    name=splited[2]
                    batch=splited[1]
                    rollno=splited[3]
                    program=splited[0]
                    dict_of_Attended[sn]=name
                    sn+=1
                print(dict_of_Attended)
            # else:
            #       dict_of_Attended["message":"No data available"]      
            return JsonResponse({'data': dict_of_Attended})

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", str(e))
            # Handle JSON decoding error
            return JsonResponse({'result': 'error', 'message': 'Invalid JSON data'}, status=400)

    return JsonResponse({'result': 'error', 'message': 'Invalid request method'}, status=400)


"""
URL configuration for tate project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home.views import *
from attendance.views import *
from facedata_collection.views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home,name="home"),
    path('home/', home,name="home"),

    path('webcam_stream/', webcam_stream, name='webcam_stream'),
    path('facedata_capture/', facedata_capture, name='facedata_capture'),
   
    path('train_model/', train_model, name='train_model'),
    path('training_model/', training_model, name='training_model'),
    
    # path('attendance/', attendance, name='attendance'),
    path('take_attendance/', take_attendance, name='take_attendance'),


    path('webcam_streamcheck/', webcam_streamcheck, name='webcam_streamcheck'),

    path('show_attended_student/', show_attended_student, name='show_attended_student'),

    path('view_attendance/', view_attendance, name='view_attendance'),
    path('showattendancebydate/', showattendancebydate, name='showattendancebydate'),
    
    path('view_students/', view_students, name='view_students'),
    path('getstudentdata/', getstudentdata, name='getstudentdata'),

]

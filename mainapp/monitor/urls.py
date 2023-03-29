from django.urls import path
from .views import *

urlpatterns = [
    path('api/upload/xray', UploadView.as_view(), name = 'prediction'),
]
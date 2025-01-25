from django.urls import path
from .views import diabetes_prediction_view

urlpatterns = [
    path('', diabetes_prediction_view, name='diabetes_prediction'),
]

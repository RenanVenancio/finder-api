from django.conf import settings
from django.urls import path, include
from person.views import MissingPersonViewSet, MissingPersonImageUpload, MissingPersonList, MissingPersonRecognize, \
    CameraCapture, TrainAlgorithm, SendAlertEmail
from rest_framework import routers
from django.conf.urls.static import static

app_name = 'person'

'''Registrando as rotas no DRF'''
router = routers.DefaultRouter()
router.register('person', MissingPersonViewSet, basename="person")
router.register('photo', MissingPersonImageUpload, basename="photo")

urlpatterns = [
                  path('api/', include(router.urls)),
                  path('api/persons', MissingPersonList.as_view(), name="personslist"),
                  path('api/photoRecognize', MissingPersonRecognize.as_view(), name="photorecognize"),
                  path('api/cameraCapture', CameraCapture.as_view(), name="cameracapture"),
                  path('api/trainAlgorithm', TrainAlgorithm.as_view(), name="trainalgorithm"),
                  path('api/sendAlertEmail', SendAlertEmail.as_view(), name="sendalertemail")
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

import django_filters
from django.http import JsonResponse
from rest_framework import viewsets, status, generics
from rest_framework.filters import SearchFilter
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from finder_admin.views import Email
from person.serializer import MissingPersonSerializer, MissingPersonSerializerPhotosSerializer, FileSerializer
from person.models import MissingPerson
from PIL import Image, ImageOps
from recognition.facial_recognition import FacialRecognition
from recognition.job import train_job


class MissingPersonViewSet(viewsets.ModelViewSet):
    """Exibindo todas as pessoas desaparecidas"""
    queryset = MissingPerson.objects.all()
    serializer_class = MissingPersonSerializer


class LargeResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100


class MissingPersonList(generics.ListAPIView):
    queryset = MissingPerson.objects.all()
    serializer_class = MissingPersonSerializer
    pagination_class = LargeResultsSetPagination
    filter_backends = [SearchFilter, django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = ['state', 'gender']
    search_fields = ['name']


class MissingPersonImageUpload(ModelViewSet):
    parser_class = (FileUploadParser,)
    serializer_class = MissingPersonSerializerPhotosSerializer


class CameraCapture(APIView):
    def get(self, request):
        recognizer = FacialRecognition()
        recognizer.webcam_capture()


class TrainAlgorithm(APIView):
    def get(self, request):
        response = {}
        train_job()
        response['message'] = 'treinamento efetuado com sucesso'
        return JsonResponse(response, safe=False)


class SendAlertEmail(APIView):
    def post(self, request):
        response = {}
        message = request.data['message']
        person_id = request.data['person_id']
        longitude = request.data['lng']
        latitude = request.data['lat']
        maps_link = 'https://maps.google.com/?q='+ str(latitude) + ',' + str(longitude)
        person = MissingPerson.objects.filter(id=person_id).last()
        html_message = '<h1>Olá, alguêm avistou  alguém parecido(a) com {}</h2>' \
                       '<p>Verifique atentamente a mensagem da pessoa que o avistou, na mesma podem ' \
                       'conter informações relevantes sobre o paradeiro da pessoa que procura.</p>' \
                       '<small>Mensagem:</small>' \
                       '<p>{}</p>' \
                       '<p>Logo abaixo está o link da localização cedido pela pessoa que enviou o alerta</p>' \
                       '<a href="{}">Localização</a>'.format(person.name, message, maps_link)
        Email.send_email(message_html=html_message,recipient_list=[person.alert_email], subject='Finder: Alerta')
        train_job()
        response['message'] = 'treinamento efetuado com sucesso'
        return JsonResponse(response, safe=False)


class MissingPersonRecognize(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        serializer = FileSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                data=serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        recognizer = FacialRecognition()
        person_details = {}
        file = Image.open(request.FILES['file'])
        file = ImageOps.exif_transpose(file)
        file = recognizer.resize_image(file, 800, 600)
        id, prec = recognizer.recognize_raw_img(file, True)
        person = MissingPerson.objects.filter(id=id)
        person = person[0]
        person_details['id'] = person.id
        person_details['name'] = person.name
        person_details['confidence'] = prec
        person_details['state'] = person.state

        # serializer = MissingPersonSerializer(person, many=True)

        return JsonResponse(person_details, safe=False)


'''
class MissingPersonImageUpload(APIView):
    parser_class = (FileUploadParser,)

    @staticmethod
    def post(request, *args, **kwargs):
        file_serializer = MissingPersonSerializerPhotosSerializer(data=request.data)

        if file_serializer.is_valid():

            file_serializer.save()

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

'''

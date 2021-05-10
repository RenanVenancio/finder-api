import django_filters
from django.http import JsonResponse
from rest_framework import viewsets, status, generics
from rest_framework.filters import SearchFilter
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib import messages
from rest_framework.views import exception_handler
from rest_framework.viewsets import ModelViewSet
import numpy as np

from person.serializer import MissingPersonSerializer, MissingPersonSerializerPhotosSerializer, FileSerializer
from person.models import MissingPerson
from PIL import Image, ImageOps

from recognition.facial_recognition import Recognizer, FacialRecognition


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


class MissingPersonRecognize(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        serializer = FileSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                data=serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        file = Image.open(request.FILES['file'])
        recognizer = FacialRecognition()
        id, prec = recognizer.recognize_raw_img(file, True)
        person = MissingPerson.objects.filter(id=id)
        serializer = MissingPersonSerializer(person, many=True)

        return JsonResponse(serializer.data, safe=False)


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

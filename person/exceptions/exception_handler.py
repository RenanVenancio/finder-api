from django.http import JsonResponse
from rest_framework.views import exception_handler

from recognition.exceptions import RecognitionError


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if isinstance(exc, RecognitionError):
        err_data = {'error': str(exc)}
        return JsonResponse(err_data, safe=False, status=400)

    return response

from rest_framework import serializers
from person.models import MissingPerson, MissingPersonPhotos


class MissingPersonSerializerPhotosSerializer(serializers.ModelSerializer):
    class Meta:
        model = MissingPersonPhotos
        fields = '__all__'


class MissingPersonSerializer(serializers.ModelSerializer):
    photos = MissingPersonSerializerPhotosSerializer(many=True, read_only=True)

    class Meta:
        model = MissingPerson
        fields = '__all__'


class FileSerializer(serializers.Serializer):
    file = serializers.FileField()
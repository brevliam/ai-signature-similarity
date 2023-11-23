from rest_framework import serializers
from .models import Signature

class SignatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Signature
        fields = ('id', 'nik', 'img', 'is_anchor')

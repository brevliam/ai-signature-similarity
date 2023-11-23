from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import SignatureSerializer
from .function import feature

class AnchorSignatureUpload(APIView):
    def post(self, request, format=None):
        try:
            request.data['is_anchor'] = True
            serializer = SignatureSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(build_result(serializer.data), status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)

class PredictSignatureSimilarity(APIView):
    def post(self, request, format=None):
        try:
            request.data['is_anchor'] = False
            serializer = SignatureSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                result = feature.predict_similarity(serializer)
                return Response(build_result(result), status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
        
def build_result(result):
    result = {
        "status": 200,
        "message": "success",
        "result": result
    }

    return result

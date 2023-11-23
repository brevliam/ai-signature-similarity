from django.apps import AppConfig
import tensorflow as tf
import os

class SignatureSimilarityConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'signature_similarity'

    model_name = 'embedding_v1_20231117.h5'
    saved_model_path = os.path.join(os.path.dirname(__file__),
                                    'model',
                                    model_name)
    loaded_model = tf.keras.models.load_model(saved_model_path)

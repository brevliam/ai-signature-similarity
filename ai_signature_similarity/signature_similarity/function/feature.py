from keras.models import Model
from keras.applications import inception_v3
from keras.metrics import CosineSimilarity
from ..libraries import utils
from ..apps import SignatureSimilarityConfig
import tensorflow as tf

class SignatureClassifier(Model):
  def __init__(self, siamese_embedding, threshold):
    super().__init__()
    self.embedding = siamese_embedding
    self.threshold = threshold

  def predict(self, data, threshold=0.86):
    similarity_score = self._compute_similarity(data)
    is_fake = similarity_score < threshold

    return {'is_fake': is_fake.numpy(),
            'similarity_score': similarity_score.numpy()}

  def _compute_similarity(self, data):
    img_ori = data[0]
    img_test = data[1]

    img_ori_emb = self.embedding(inception_v3.preprocess_input(img_ori))
    img_test_emb = self.embedding(inception_v3.preprocess_input(img_test))

    cos_similarity = CosineSimilarity()
    similarity_score = cos_similarity(img_ori_emb, img_test_emb)

    return similarity_score

def predict_similarity(serializer):
    nik = serializer.data.get('nik')
    anchor_path, test_path = utils.find_saved_signatures_by_nik(nik)
    anchor_image = preprocess_image(anchor_path)
    test_image = preprocess_image(test_path)
    
    data = (anchor_image, test_image)
    emb_model = SignatureSimilarityConfig.loaded_model
    model = SignatureClassifier(emb_model, 0.86)
    result = model.predict(data)

    utils.delete_signature_data_by_nik(nik)

    return result
  
def preprocess_image(image_path):
    target_shape = (200, 200)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.expand_dims(image, axis=0)

    return image

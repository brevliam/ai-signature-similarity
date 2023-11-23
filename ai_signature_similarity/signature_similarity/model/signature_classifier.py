from keras.metrics import CosineSimilarity
from keras.applications import inception_v3
from keras.models import Model
import tensorflow as tf
import os

saved_model_path = os.path.join(os.path.dirname(__file__),
                                'embedding_v1_20231117.h5')
emb_model = tf.keras.models.load_model(saved_model_path)

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

    img_ori_emb = emb_model(inception_v3.preprocess_input(img_ori))
    img_test_emb = emb_model(inception_v3.preprocess_input(img_test))

    cos_similarity = CosineSimilarity()
    similarity_score = cos_similarity(img_ori_emb, img_test_emb)

    return similarity_score
  
def preprocess_image(image):
    """
    Load the specified file as a PNG image, preprocess it and
    resize it to the target shape.
    """
    target_shape = (200, 200)
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.expand_dims(image, axis=0)

    return image

def predict_similarity(anchor, test):
    anchor_path = anchor.img.path
    test_path = test.img.path

    anchor_image = preprocess_image(anchor_path)
    test_image = preprocess_image(test_path)
    data = (anchor_image, test_image)

    model = SignatureClassifier(emb_model, 0.86)
    result = model.predict(data)

    return result
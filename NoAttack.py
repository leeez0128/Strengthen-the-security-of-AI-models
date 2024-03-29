import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from art.estimators.classification import KerasClassifier
from tensorflow.python.framework.ops import disable_eager_execution

tf.compat.v1.experimental.output_all_intermediates(True)

def runNoAttack():
    disable_eager_execution()
    img = image.load_img('./Siamese_cat.jpeg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = ResNet50(weights='imagenet')

    classifier = KerasClassifier(model=model)
    attacker = None
    adversarial_image = img

    prediction = classifier.predict(adversarial_image)
    result = decode_predictions(prediction, top=5)
    print('====== RUN with NO ATTACK ======')
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')


if __name__ == '__main__':
    runNoAttack()
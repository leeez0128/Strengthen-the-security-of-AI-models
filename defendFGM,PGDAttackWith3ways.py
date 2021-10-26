import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.preprocessor import GaussianAugmentation, SpatialSmoothing, FeatureSqueezing

from art.utils import to_categorical

from tensorflow.python.framework.ops import disable_eager_execution

def runFastGradientMethod():
    disable_eager_execution()
    img = image.load_img('./Siamese_cat.jpeg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = ResNet50(weights='imagenet')

    classifier = KerasClassifier(model=model, clip_values=(0,255))
    
    # med -> eps = 4
    attacker = FastGradientMethod(estimator=classifier, eps=4)
    adversarial_image = attacker.generate(img)

    print('\n>>>>> GENERATING ADVERSARIAL_IMAGE WITH FastGradientMethod')
    adversarial_image = attacker.generate(img)

    print('\n>>>>> Defend Attack with Gaussian Noise, Spatial Smoothing, Feature Squeezing')
    # Gaussian Noise
    filtered_adversarial_image, _ = GaussianAugmentation(sigma=9, augmentation=False)(adversarial_image)
    # Spatial Smoothing
    filtered_adversarial_image, _ = SpatialSmoothing(window_size = 7)(filtered_adversarial_image)
    # Feature Squeezing
    filtered_adversarial_image, _ = FeatureSqueezing(bit_depth=3, clip_values=(0, 255))(filtered_adversarial_image)
    
    prediction = classifier.predict(filtered_adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')


def runProjectedGradientDescent():
    disable_eager_execution()
    img = image.load_img('./Siamese_cat.jpeg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = ResNet50(weights='imagenet')

    classifier = KerasClassifier(model=model, clip_values=(0,255))
    
    attacker = ProjectedGradientDescent(estimator=classifier, targeted=True, eps_step=1, eps=8, max_iter=3)
    y = to_categorical([430],1000) #traget class = basketball


    print('\n>>>>> GENERATING ADVERSARIAL_IMAGE WITH ProjectedGradientDescent')
    adversarial_image = attacker.generate(img, y=y)

    print('\n>>>>> Defend Attack with Gaussian Noise, Spatial Smoothing, Feature Squeezing')
    # Gaussian Noise
    filtered_adversarial_image, _ = GaussianAugmentation(sigma=9, augmentation=False)(adversarial_image)
    # Spatial Smoothing
    filtered_adversarial_image, _ = SpatialSmoothing(window_size = 7)(filtered_adversarial_image)
    # Feature Squeezing
    filtered_adversarial_image, _ = FeatureSqueezing(bit_depth=3, clip_values=(0, 255))(filtered_adversarial_image)
    
    prediction = classifier.predict(filtered_adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')

    
if __name__ == '__main__':
    runFastGradientMethod()
    print()
    runProjectedGradientDescent()

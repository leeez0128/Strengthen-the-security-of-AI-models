import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.preprocessor import GaussianAugmentation, SpatialSmoothing, FeatureSqueezing

from tensorflow.python.framework.ops import disable_eager_execution

def runFastGradientMethod():
    disable_eager_execution()
    img = image.load_img('./Siamese_cat.jpeg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = ResNet50(weights='imagenet')

    classifier = KerasClassifier(model=model)
    
    # med -> eps = 4
    attacker = FastGradientMethod(estimator=classifier, eps=4)
    
    print('\n>>>>> GENERATING ADVERSARIAL_IMAGE WITH FastGradientMethod')
    adversarial_image = attacker.generate(img)

    print('\n>>>>> NO DEFENSE')
    prediction = classifier.predict(adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')


    # Gaussian Noise
    gaussian_noise_adversarial_image, _ = GaussianAugmentation(sigma=9, augmentation=False)(adversarial_image)
    print('\n>>>>> Defend Attack with gaussian noise')
    prediction = classifier.predict(gaussian_noise_adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')

    # Spatial Smoothing
    spatial_smoothing_adversarial_image, _ = SpatialSmoothing(window_size = 7)(adversarial_image)
    print('\n>>>>> Defend Attack with spatial smoothing')
    prediction = classifier.predict(spatial_smoothing_adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')


    # Feature Squeezing
    feature_squeezing_adversarial_image, _ = FeatureSqueezing(bit_depth=3, clip_values=(0, 255))(adversarial_image)
    print('\n>>>>> Defend Attack with feature squeezing')
    prediction = classifier.predict(feature_squeezing_adversarial_image)
    result = decode_predictions(prediction, top=5)
    for idx, value in enumerate(result[0]) :
        print(f'{idx+1}. {value[1]} with a {value[2]*100 : .5f}%')


    
if __name__ == '__main__':
    runFastGradientMethod()

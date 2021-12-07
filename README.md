# ART - IBM Research
## Precondition
```
$ source ~/.bash_profile
$ conda activate venv
```

## Install
```
$ pip install tensorflow
$ pip install adversarial-robustness-toolbox[keras]
$ pip install pillow
```

## Run
```
$ python NoAttack.py  # AI모델에 대해 아무런 공격을 하지 않음
$ python FastGradientMethod.py  # AI모델에 대해 Fast Gradient Method 공격
$ python ProjectedGradientDescent # AI모델에 대해 Projected Gradient Descent 공격
$ python defendFGM,PGDAttackWith3ways.py #AI 모델에 FGM, PGD 공격 및 3가지 방어기법 사용
```
#### ⭐️AI모델에 대해 Projected Gradient Descent 공격
1. targeted = True <br>
2. attcker generate 시 y값으로 snail 혼동 추가 <br>
3. stop 이미지는 snail로 오인하게 되는 공격 받음 <br>
4. 방어 기법 종류 : guassian noise, spatial smoothing, feature squeezing <br>
  - 방어 기법 1개씩 사용
  - 방어 기법 2개씩 사용
  - 방어 기법 3개 모두 사용
```
$ python YoloTest.py
```

## Results
|| NoAttack | DefendWith3ways |
|--------|--------|--------|
| RESULT | <img width="230" alt="스크린샷 2021-10-24 오후 9 38 15" src="https://user-images.githubusercontent.com/48276633/138594545-688b09a0-0d96-4186-9fbd-6fa86f3a960c.png"> | <img width="450" alt="DefendWith3ways" src="https://user-images.githubusercontent.com/48276633/139355998-e9eb5c72-55a5-41b5-b196-cc4a21be625c.png">|

|| FastGradientMethod | ProjectedGradientDescent |
|--------|--------|--------|
| RESULT | <img width="310" alt="FastGradientMethod png" src="https://user-images.githubusercontent.com/48276633/139356695-714c112d-19dd-4519-8ef7-623d2fdcc795.png"> | <img width="370" alt="ProjectedGradientDescent" src="https://user-images.githubusercontent.com/48276633/139356697-fc3052e0-cfe9-4a42-a679-f6f9815a304b.png">|

<br><br>

# [Physical Adversarial Examples Against Deep Neural Networks](https://bair.berkeley.edu/blog/2017/12/30/yolo-attack/)
Recent research has shown that DNNs are vulnerable to adversarial examples: Adding carefully crafted adversarial perturbations to the inputs can mislead the target DNN into mislabeling them during run time.
### - Digital Adversarial Examples
Optimization based methods have also been proposed to create adversarial perturbations for targeted attacks. Specifically, these attacks formulate an objective function whose solution seeks to maximize the difference between the true labeling of an input, and the attacker’s desired target labeling, while minimizing how different the inputs are, for some definition of input similarity.

### - Physical Adversarial Examples
Computer vision algorithms identify relevant objects in a scene and predict bounding boxes indicating objects’ position and kind. Compared with classifiers, detectors are more challenging to fool as they process the entire image and can use contextual information (e.g. the orientation and position of the target object in the scene) in their predictions.

<hr>

* Attack Strength, Deffense Strength
Attack Strength : medium <br>
Deffend Strength : high <br>

* https://github.com/Trusted-AI/adversarial-robustness-toolbox


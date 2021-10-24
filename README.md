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
$ python Projected Gradient Descent # AI모델에 대해 Projected Gradient Descent 공격
```

## Results
| | NoAttack | FastGradientMethod | ProjectedGradientDescent |
|--------|--------|--------|--------|
| Results | <img width="230" alt="스크린샷 2021-10-24 오후 9 38 15" src="https://user-images.githubusercontent.com/48276633/138594545-688b09a0-0d96-4186-9fbd-6fa86f3a960c.png"> | <img width="280" alt="스크린샷 2021-10-24 오후 9 38 33" src="https://user-images.githubusercontent.com/48276633/138594578-98d0d09b-3334-4a60-b820-16a4f8f9cef2.png"> |<img width="280" alt="스크린샷 2021-10-24 오후 9 38 55" src="https://user-images.githubusercontent.com/48276633/138594587-78c02368-c4e3-4e7f-a5de-9849b181c150.png">


<hr>

* Attack Strength, Deffense Strength
Attack Strength : medium <br>
Deffend Strength : high <br>


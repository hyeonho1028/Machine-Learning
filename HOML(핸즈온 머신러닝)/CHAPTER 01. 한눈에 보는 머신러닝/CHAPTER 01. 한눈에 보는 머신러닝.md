핸즈온 머신러닝



# 한눈에 보는 머신러닝

수억명의 생활을 편리하게 만들어 주류가 된 첫 번째 머신러닝 애플리케이션은 1990년대에 시작된 스팸필터이다. 스스로 생각하는 스카이넷정도는 아니지만 기술적으로 머신러닝이라 할  수 있습니다.



이 장에서는 머신러닝의 그림을 조망하고 주요 영역과 가장 중요한 랜드마크인 지도 학습과 비지도 학습, 온라인 학습과 배치 학습, 사례기반 학습과 모델 기반 학습을 알아보겠습니다. 또한 전형적인 머신러닝 프로젝트의 작업 흐름을 살펴보고 만날 수 있는 주요 문제점과 머신러닝 시스템을 평가하고 세밀하게 튜닝하는 방법을 다루겠습니다.



## 1.1 머신러닝이란?

머신러닝은 데이터로부터 학습하도록 컴퓨터를 프로그래밍하는 과학(또는 예술)입니다.



조금더 일반적으로 정의한다면

명시적인 프로그래밍 없이 컴퓨터가 학습하는 능력을 갖추게 하는 연구 분야이다.(아서 사무엘)



공학적으로 정의한다면

어떤 작업 T에 대한 컴퓨터 프로그램의 성능을 P로 측정했을 떄 경험 E로 인해 성능이 향상됐다면, 이 컴퓨터 프로그램은 작업 T와 성능 측정 P에 대해 경험 E로 학습한 것이다.(톰 미첼)



스팸필터의 사례에서

시스템이 학습하는 데 사용하는 샘플을 훈련 세트 : training set

각 훈련 데이터를 훈련 사례 : training instance

작업 T란 새로운 메일이 스팸인지 구분하는 것이고, 경험 E는 training data이며, 성능 측정 P는 직접 정의해야 합니다. 예를 들어 정확히 분류된 메일의 비율을 P로 사용할 수 있습니다. 이 성능 측정을 흔히 accuarcy 라고 부릅니다.



## 1.2 왜 머신러닝을 사용하는가?

전통적인 프로그래밍 기법을 사용해 스팸 필터를 만들 수 있을지 생각해 봅시다.

스팸에 어떤 단어들이 주로 나타나는지 살펴보고, 빈번하게 나타나는 단어들에 대해 각 패턴을 감지하는 알고리즘을 작성하여 프로그램이 이러한 패턴을 발견햇을 떄 그 메일을 스팸으로 분류합니다.

충분한 성능이 나올 때까지 위의 과정을 반복합니다.



반면에 머신러닝 기법에 기반을 둔 스팸 필터는 일반 메일에 비해 스팸에 자주 나타나는 패턴을 감지하여 어떤 단어와 구절이 스팸 메일을 판단하는 데 좋은 기준인지 자동으로 학습합니다. 그러므로 프로그램이 훨씬 짧아지고 유지 보수하기 쉬우며 대부분 정확도가 더 높습니다.



머신러닝이 유용한 또 다른 분야는 전통적인 방식으로는 너무 복잡하거나 알려진 알고리즘이 없는 문제입니다. 예를 들어 음성인식(speech recognition)입니다. 수백만명이 말하는 여러 언어로 된 수천개의 단어를 구분하는 것으로 확장하여 하드코딩하는 것은 매우 어렵습니다. 그렇기 때문에 각 단어를 녹음한 샘플을 사용해 스스로 학습하는 알고리즘을 작성하는 것이 현재 가장 큰 솔루션입니다. (전통적인 방법은 'one', 'two'을 구분할 때 사운드 'T'로 시작하는 높은 피치의 사운드 강도를 측정하는 알고리즘을 하드코딩 해야 한다.)



또한 우리는 머신러닝을 통해 배울 수도 있습니다.

스팸메일로 돌아와 스팸을 예측하기 가장 좋은 패턴(단어와 단어의 조합)을 확인 할 수 있는데, 이 때 예상치 못한 연관 관계나 새로운 추세가 발견되기 합니다. 이를 통해 해당 문제를 더 잘 이해하도록 도와줍니다.



위와 같이 머신러닝 기술을 적용해서 대용량의 데이터를 분석하면 겉으로는 보이지 않던 패턴을 발견할 수 있습니다. 이를 데이터 마이닝 이라고 합니다.



머신러닝이 뛰어난 분야

> 기존 솔루션으로는 많은 수동 조정과 규칙이 필요한 문제 : 하나의 머신러닝 모델이 코드를 간단하고 더 잘 수행되도록 할 수 있습니다.
>
> 전통적인 방식으로는 전혀 해결 방법이 없는 복잡한 문제 : 가장 뛰어난 머신러닝 기법으로 해결 방법을 찾을 수 있습니다.
>
> 유동적인 환경 : 머신러닝 시스템은 새로운 데이터에 적응할 수 있습니다.
>
> 복잡한 문제와 대량의 데이터에서 통찰 얻기



## 1.3 머신러닝 시스템의 종류

머신러닝은 굉장히 포괄적인 개념입니다. 또한 시스템의 종류가 많으므로 넓은 범주에서 분류할 수 있습니다.

> 사람의 감독 하에 훈련하는 것인지 그렇지 않은 것인지(지도, 비지도, 준지도, 강화 학습)
>
> 실시간으로 점진적인 학습을 하는지 아닌지(온라인 학습과 배치 학습)
>
> 단순하게 알고 있는 데이터 포인트와 새 데이터 포인트를 비교하는 것인지 아니면 훈련 데이터셋에서 과학자들처럼 패턴을 발견하여 예측 모델을 만드는지(사례 기반 학습과 모델 기반 학습)

중요한 것은 위의 범주가 서로 배타적이지 않고 원하는 대로 연결 할 수 있다는 점입니다. 예를 들어 최첨단 스팸필터가 심층 신경망 모델을 사용해 스팸과 스팸이 아닌 메일로부터 실시간으로 학습할 지도 모릅니다. 그렇다면 이 시스템은 온라인이고 모델 기반이며 지도 학습 시스템입니다.



### 1.3.1 지도 학습과 비지도 학습














































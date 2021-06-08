# FACNN

1. 주안점
    - 가벼운 모델일 것
    - 엣지를 잘 살리는 모델일 것

2. 시도한 것
    - BSRGAN에서 사용하는 degradation 일부 차용
    - Edge 검출인 Canny를 활용하여 검출한 부분을 씌워서 degradation
    ![](canny.png)

3. 실험결과
    - degradation 부분은 모델이 깊지 않아서 그런지 크게 영향을 미치지 못한 것 같다.
    - canny degradation 방법은 오히려 엣지를 뭉개버리는 참사가 일어났다.
    ![](snapshot.jpg)
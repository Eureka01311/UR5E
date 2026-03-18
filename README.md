# UR5e_Vision_Pick_and_Place

MuJoCo 시뮬레이터와 OpenCV를 활용한 UR5e 로봇 팔 객체 인식 및 Pick and Place 프로젝트입니다.

## 1. 프로젝트 결과물

| 시뮬레이션 환경 및 로봇 제어 (MuJoCo) | 객체 인식 및 좌표 산출 (OpenCV) |
| :---: | :---: |
| ![grab](https://github.com/user-attachments/assets/037d01d3-77d9-469d-8abd-e49d2563a20a) | ![vision](https://github.com/user-attachments/assets/60b90af0-103b-42f9-a590-726d5a7a062d) |

---

## 2. 프로젝트 개요 (Overview)

* **목적:** 카메라 데이터를 활용하여 목표 객체(사과)를 인식하고, 3D 공간 좌표로 변환한 뒤 로봇 팔을 제어하여 목표 지점(상자)으로 옮기는 파지 및 적재(Pick and Place) 알고리즘 구현.
* **환경:** Python, MuJoCo Physics Engine, OpenCV 4.x.

---

## 3. 알고리즘 설명 (Algorithm)

* **Object Detection:** OpenCV를 사용하여 카메라 이미지에서 대상 객체(사과)의 특징을 추출하고, 중심점을 기준으로 Bounding Box를 생성합니다.
* **2D to 3D Coordinate Transformation:** 카메라의 FOV(Field of View)와 삼각비를 활용합니다. 2D 이미지의 픽셀 오차와 픽셀 거리를 분석하여 실제 시뮬레이션 공간의 3D 직교 좌표(x, y, z)로 변환합니다.
* **Kinematics (운동학 해석):**
  * **Forward Kinematics:** 현재 로봇의 Joint States를 기반으로 엔드 이펙터(End-effector)의 현재 위치를 계산합니다.
  * **Inverse Kinematics:** 비전 알고리즘으로 산출된 3D 목표 좌표에 도달하기 위해 필요한 6축 관절(Shoulder, Arm, Wrist)의 목표 각도를 역산합니다.
* **Actuation Control:** 계산된 Joint 각도를 MuJoCo 환경의 모터 액추에이터 제어값으로 인가하여 로봇의 움직임을 구현합니다.

---

## 4. 주의사항 및 한계점 (Limitations)

> **[중요] 시각 기반 좌표 변환의 제한성**
> * 현재 알고리즘은 고정된 카메라 뷰와 설정된 FOV 값에 기반한 삼각비 연산에 최적화되어 있습니다.
> * 카메라의 위치나 해상도가 변경될 경우, 픽셀-물리거리 변환 비율(Ratio) 및 캘리브레이션 파라미터의 전면적인 재조정이 필요합니다.
> * 단일 카메라 이미지의 픽셀 밀도에 의존하므로, 원근에 따른 Z축(Depth) 오차가 발생할 수 있습니다.

---

## 5. 실행 방법 (Usage)

**의존성 패키지 설치**
```bash
pip install mujoco opencv-python numpy
```

**실행 명령어**
```bash
# 터미널 1
python3 ur5e_main_V1.0.py

# 터미널 2
python3 ur5e_control_V1.0.py
```

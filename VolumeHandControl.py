import cv2
import numpy
import HandTrackingModule
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# pega as medidas de volume do computador
# no meu caso o valor minimo é -65 e o maximo 0
volume_range = volume.GetVolumeRange()
# cria alguma variaveis auxiliares
min_volume = volume_range[0]
max_volume = volume_range[1]
vol_bar = 400
vol_percentage = 0

# pega a camera
cap = cv2.VideoCapture(0)

detector = HandTrackingModule.HandDetector(min_detection_confidence=0.7)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    hand_points = detector.find_points(img)
    if len(hand_points) != 0:
        # pega as posicoes do dedo indicador e polegar
        # para saber qual posicao representa cada dedo acesse https://google.github.io/mediapipe/solutions/hands.html
        index_finger_x, index_finger_y = hand_points[4][1], hand_points[4][2]
        thumb_x, thumb_y = hand_points[8][1], hand_points[8][2]
        # pega a posicao centrar entre os dois dedos
        center_x, center_y = (index_finger_x + thumb_x)//2, (index_finger_y + thumb_y)//2

        # desenha ponto nos dedos que serao usados para o volume
        cv2.circle(img, (index_finger_x, index_finger_y), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
        # desenha uma linha ligando ambos os pontos
        cv2.line(img, (index_finger_x, index_finger_y), (thumb_x, thumb_y), (255, 0, 0), 3)
        # desenha um ponto no centro da linha
        cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)

        fingers = detector.fingers_up()
        # se o dedo midinho abaixar define o volume
        if not fingers[4]:
            length = math.hypot(thumb_x - index_finger_x, thumb_y - index_finger_y)

            # pega o tamanho da linha entre os dedos
            # 30 - 220
            # esses valores sao a distancia minima(dedos encostados) e maxima(dedos totalmente abertos) que os dedos podem ter
            # podem mudar dependendo da distancia da mao
            # faz as conversoes. EX: 30 equivale a -65(min_volume) e 220 a 0(max_volume)
            vol = numpy.interp(length, [30, 220], [min_volume, max_volume])
            vol_bar = numpy.interp(length, [30, 220], [400, 150])
            vol_percentage = numpy.interp(length, [30, 220], [0, 100])
            # define o volume do computador
            volume.SetMasterVolumeLevel(vol, None)

    # faz os desenhos dos retangulos e a porcentagem na tela
    cv2.rectangle(img, (50, 150), (75, 400), (0, 0, 255), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (75, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), )

    # mostra a imagem da camera
    cv2.imshow('Camera', img)
    cv2.waitKey(1)

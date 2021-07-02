import cv2
import numpy as np
import os
import HandTrackingModule

# espessura do pincel
brush_thickness = 15
# espessura da borracha
eraser_thickness = 50

folder_path = "header"
# pega as imagens da pasta header
myList = os.listdir(folder_path)

# adiciona as imagens da pasta a uma lista
header_list = []
for img_path in myList:
    image = cv2.imread(f'{folder_path}/{img_path}')
    header_list.append(image)

# mostra a imagem da posicao 0 inicialmente
header = header_list[0]

# pega a camera //comentar sobre a comera
cap = cv2.VideoCapture(1)
# define o tamanho da tela 3=width e 4=height
cap.set(3, 1280)
cap.set(4, 720)

detector = HandTrackingModule.HandDetector(min_detection_confidence=0.65, max_num_hands=1)

selected_color = (255, 0, 255)

img_paint = np.zeros((720, 1280, 3), np.uint8)
# ponto inicial da linha a ser desenhada
x_previous, y_previous = 0, 0

while True:
    success, img = cap.read()
    # inverte a imagem da camera
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    hand_points = detector.find_points(img)
    if len(hand_points) != 0:
        # pega as posicoes do dedo indicador e do dedo do meio
        # para saber qual posicao representa cada dedo acesse https://google.github.io/mediapipe/solutions/hands.html
        index_finger_x, index_finger_y = hand_points[8][1], hand_points[8][2]
        middle_finger_x, middle_finger_y = hand_points[12][1], hand_points[12][2]

        fingers = detector.fingers_up()
        # se o dedo indicador e dedo do meio estiverem erguidos usa o modo de selecao
        if fingers[1] and fingers[2]:
            x_previous, y_previous = 0, 0
            # se a posicao y do dedo indicador estiver no header
            if index_finger_y < 125:
                # se a posicao x do dedo indicador estiver em alguma dessas posicoes
                # troca o header, para o que indica a cor selecionada, e tambem troca a cor
                if 250 < index_finger_x < 450:
                    header = header_list[0]
                    selected_color = (255, 0, 255)
                elif 550 < index_finger_x < 750:
                    header = header_list[1]
                    selected_color = (255, 0, 0)
                elif 800 < index_finger_x < 950:
                    header = header_list[2]
                    selected_color = (0, 255, 0)
                elif 1050 < index_finger_x < 1200:
                    header = header_list[3]
                    selected_color = (0, 0, 0)

        # se o dedo indicador estiver erguido e o do meio abaixado usa o modo de desenho
        if fingers[1] and not fingers[2]:
            # desenha um circulo na ponta do dedo indicador
            cv2.circle(img, (index_finger_x, index_finger_y), 15, selected_color, cv2.FILLED)

            # verificacao para que ele nao desenhe uma linha vinda do canto da tela
            if x_previous == 0 and y_previous == 0:
                x_previous, y_previous = index_finger_x, index_finger_y

            # variavel para verificacao se o tipo selecionado nao é a borracha
            is_eraser = selected_color == (0, 0, 0)
            # desenha uma linha do ponto inicial ate o dedo indicador
            # como essa variavel sera sempre zerada em cada renderizacao ele ira desenhar pontos que se conectam nao linhas
            cv2.line(img, (x_previous, y_previous), (index_finger_x, index_finger_y),
                     selected_color, eraser_thickness if is_eraser else brush_thickness)
            cv2.line(img_paint, (x_previous, y_previous), (index_finger_x, index_finger_y),
                     selected_color, eraser_thickness if is_eraser else brush_thickness)
            x_previous, y_previous = index_finger_x, index_finger_y

    # converte a imagem para escala de cinza
    img_gray = cv2.cvtColor(img_paint, cv2.COLOR_BGR2GRAY)
    # pega a imagem em escala de cinza e separa o desenho do background
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # transforma o objeto separado em uma imagem BGR
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    # faz a uniao entre as imagens separadas
    # https://www.pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_paint)

    # define a posicao da imagem do header
    # pode ocorrer algum problema com a resolucao da camera, tera de ser ajustado a resolucao para uma correspondente
    # ou usar uma imagem de tamanho diferente
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, img_paint, 0.5, 0)
    cv2.imshow("Camera", img)
    # cv2.imshow("Paint", img_paint)
    # cv2.imshow("Inverse", img_inverse)
    cv2.waitKey(1)


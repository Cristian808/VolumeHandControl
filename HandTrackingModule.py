# caso algum problema com dll quando for usar o mediapipe acesse https://github.com/google/mediapipe/issues/1839
import cv2
import mediapipe


class HandDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # auxiliares para acesso a funcoes do mediapipe
        self.mediapipe_hands = mediapipe.solutions.hands
        self.mediapipe_draw = mediapipe.solutions.drawing_utils

        # pega as maos e armazena em uma lista
        self.hands = self.mediapipe_hands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        # variaveis auxiliares
        self.hand_points_position = []
        self.process_result = None
        # variavel para armazenar a posicao do ponto na extremidade de cada dedo(4=dedao, 8=indicador, etc)
        self.finger_points  = [4, 8, 12, 16, 20]

    def find_hands(self, img):
        # converte a imagem da camera
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # processa a imagem enviada e retorna os pontos de referencia de cada mao
        self.process_result = self.hands.process(image_rgb)
        if self.process_result.multi_hand_landmarks:
            for landmark in self.process_result.multi_hand_landmarks:
                # para cada ponto da mao desenha o ponto e as conexoes
                self.mediapipe_draw.draw_landmarks(img, landmark, self.mediapipe_hands.HAND_CONNECTIONS)

        # retorna a imagem processada
        return img

    def find_points(self, img, hand_number=0):
        self.hand_points_position = []
        if self.process_result.multi_hand_landmarks:
            # pega os pontos da mao especificada pela variavel hand_number
            my_hand = self.process_result.multi_hand_landmarks[hand_number]
            for id, hand_point in enumerate(my_hand.landmark):
                # para cada ponto faz o calculo para saber qual posicao do pixel na tela
                height, width, channels = img.shape
                pixel_x, pixel_y = int(hand_point.x * width), int(hand_point.y * height)
                self.hand_points_position.append([id, pixel_x, pixel_y])

        # retorna a lista com o id de cada ponto e suas posicoes na tela
        return self.hand_points_position

    def fingers_up(self):
        fingers = []
        # se o ponto da ponta do dedao estiver em uma posicao x > x da posicao do ponto abaixo deste o dedos estara erguido
        if(self.hand_points_position[self.finger_points[0]][1] >
                self.hand_points_position[self.finger_points[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        # para os outros quatro dedos ele pegara a posicao y da ponta do dedo e verificara se ela esta < que dois pontos abaixo da ponta do dedo
        for finger in range(1, 5):
            if(self.hand_points_position[self.finger_points[finger]][2]
                    < self.hand_points_position[self.finger_points[finger] - 2][2]):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

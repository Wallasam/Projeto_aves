import serial
import cv2
import time
import threading
from ultralytics import YOLO


#Conexão com Arduino
conect = serial.Serial('COM4', 9600) #Conectamos a porta desejada e setamos o baudrate (Quantidade de informações por segundo será passado)

def ativar_sistema():
    conect.write(b'1')
    print("--- O Sistema está ATIVO! ---")
    time.sleep(5)

#Modelo utilizado
model = YOLO('runs/detect/train6/weights/best.pt')

#Iniciando as variáveis de câmera e área proibida
camCap = cv2.VideoCapture(0)
areaLimit = [500, 300, 1000, 715] #Localização da Área proibida para pássaros
                                  #Os 2 Primeiros valores são os Pontos Iniciais X e Y e os últimos dois são os Pontos Finais X e Y

while True:
    check, img = camCap.read()
    img = cv2.resize(img, (1270, 720))

    imgArea = img.copy()
    cv2.rectangle(imgArea, (areaLimit[0], areaLimit[1]), (areaLimit[2], areaLimit[3]), (235, 100, 200), -1)

    resultado = model(img) #Informações sobre a imagem

    for objetos in resultado: #Vamos percorrer as informações presentes nos resultados
        obj = objetos.boxes #Atribuimos as inf. de boxes a uma variável

        for dados in obj: #Filtrando as informações necessárias sobre posicionamento do pássaro

            x, y, w, h = dados.xyxy[0]
            x, y, w, h = int(x), int(y), int(w), int(h)

            cls = int(dados.cls[0]) #Extraindo apenas a classx
            centerX, centerY = (x+w)//2, (y+h)//2 #Localizando o Centro do Objeto (Pássaro)

            if cls == 1: #Se a classe identificada foi igual ao valor da classe de pássaro, trackeamos a imagem
                cv2.rectangle(img, (x, y), (w, h), (255, 0, 255), 3)  # Criamos um retângulo para trackear o animal detectado, temos os parâmetros de Imagem detectada, localização X e Y, Localização W e H, Cor e Espessura
                cv2.circle(img, (centerX, centerY), 5, (0, 0, 0), 5) # Trackeando centro do OBJ

                #Criando Condição para não ultrapassar a área
                if centerX >= areaLimit[0] and centerX <= areaLimit[2] and centerY >= areaLimit[1] and centerY <= areaLimit[3]:
                    print('Dentro da área')
                    #Texto de Alerta
                    cv2.putText(img, 'Bird Detected', ( 100, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

                    #Ativando Sistema de Dispersão
                    ativar_sistema()

    #Diminuindo a Opacidade da imagem para melhor visualização
    finalImg = cv2.addWeighted(imgArea, 0.3, img, 0.3, 0)
    cv2.imshow('img', finalImg)

    # Sistema de dispersão inativo
    conect.write(b'0')
    print("--- O Sistema está desativado! ---")

    k = cv2.waitKey(1)
    if k == ord('q'):
        conect.write(b'0')
        break
from ultralytics import YOLO

def main():

    #Iniciando o treino das imagens utilizando o modelo Yolov8s
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    # Usamos a instância de treino para começar
    # Dentro desta instância, colocamos alguns parâmetros para melhorar o treinamento
    #   data: arquivo .yaml onde localizamos as pastas que estão localizadas as imagens e criamos as classes
    #   epochs: A quantidade de vezes que o treino passará por todas as imagens/ device: utilizamos uma GPU para realizar o treino / Batch: tamanho do lote que será usado para treino
    #   lr0: Learning Rate ou Taxa de aprendizado/ mosaic: Data Augmentation ou Aumento de Dados que serão utilizados

    model.train(data="C:/Users/dedei/PycharmProjects/birds_detection/datasets/data.yaml", epochs=40, device=0, batch=16, lr0=0.1, mosaic=1.0, box=0.05 )  # train the model
    metrics = model.val()  # Valor da performance do modelo



if __name__ == '__main__':
    main()
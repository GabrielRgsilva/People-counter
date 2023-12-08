import cv2
import numpy as np

# carregar arquivo de vídeo
cap = cv2.VideoCapture('input.mp4')
count_line_position = 499  # Posição da linha de contagem

# Configurações mínimas para detecção de objetos
min_width = 50
min_height = 50

# Criação do algoritmo para subtração de fundo
algo = cv2.createBackgroundSubtractorMOG2()

# Função para calcular o centro de um retângulo
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []  # Lista para armazenar os centros dos objetos detectados
offset = 4  # Margem de aceitação ao cruzar a linha de contagem
counter = 0  # Contador de objetos que cruzaram a linha

# Inicializar o VideoWriter para gravar o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (800, 600))  # Nome do arquivo e configurações de gravação

while True:
    ret, frame1 = cap.read()  # Lê cada frame do vídeo
    frame1 = cv2.resize(frame1, (800, 600))  # Redimensiona o frame para uma resolução específica
    
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Converte o frame para tons de cinza

    if 'bgframe' not in locals():
        bgframe = gray.copy()  # Define o primeiro frame como plano de fundo

    frameDelta = cv2.absdiff(bgframe, gray)  # Calcula a diferença entre o plano de fundo e o frame atual

    _, thresh = cv2.threshold(frameDelta, 70, 100, cv2.THRESH_BINARY)  # Define um limite para a diferença calculada

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, None, iterations=4)  # Aplica uma operação de dilatação para melhorar as formas

    # Encontra contornos dos objetos na imagem
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha a linha de contagem no frame
    cv2.line(frame1, (25, count_line_position), (1800, count_line_position), (255, 127, 0))

    # Percorre os contornos encontrados
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Obtém as coordenadas do retângulo envolvente do contorno
        if w >= min_width and h >= min_height:  # Verifica se atende aos requisitos de tamanho
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha um retângulo ao redor do objeto
            
            center = center_handle(x, y, w, h)  # Calcula o centro do retângulo
            detect.append(center)  # Adiciona o centro à lista de detecções
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)  # Desenha um círculo no centro do objeto
            
            for cx, cy in detect:
                if count_line_position - offset < cy < count_line_position + offset:
                    counter += 1  # Incrementa o contador se o objeto cruzar a linha de contagem
                    detect.remove((cx, cy))  # Remove o centro do objeto da lista após cruzar a linha
            
            cv2.line(frame1, (25, count_line_position), (1800, count_line_position), (0, 127, 255), 3)  # Linha de contagem
            print("Counter: " + str(counter))  # Exibe o contador na saída
            
    cv2.putText(frame1, "Counter: " + str(counter), (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)  # Texto do contador

    cv2.imshow('video original', frame1)  # Mostra o vídeo original com as detecções
    
    output.write(frame1)  # Escreve o frame processado no arquivo de vídeo de saída

    if cv2.waitKey(1) == 13:  # Verifica se a tecla Enter foi pressionada para encerrar o loop
        break

# Libera recursos e fecha as janelas
output.release()
cv2.destroyAllWindows()
cap.release()
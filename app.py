import cv2
import pytesseract

def encontrarRoiPlaca(source):
    img = cv2.imread(source)
    #cv2.imshow("img", img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("cinza", img)

    #_, bin = cv2.threshold(cinza, 128, 255, cv2.THRESH_BINARY)
    _, bin = cv2.threshold(cinza, 216, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", img)

    desfoque = cv2.GaussianBlur(bin, (5, 5), 0)
    cv2.imshow("defoque", desfoque)

    contornos, hierarquia = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contornos, -1, (0, 255, 0), 1)

    for idx, c in enumerate(contornos):
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                roi = img[y:y + lar, x:x + alt]
                cv2.imwrite(f"output/roi{idx}.png", roi)

    cv2.imshow("contornos", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preProcessamentoRoiPlaca(source):
    img_roi = cv2.imread(source)

    if img_roi is None:
        return

    resize_img_roi = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

    # Binariza imagem
    _, img_binary = cv2.threshold(img_cinza, 70, 255, cv2.THRESH_BINARY)

    # Desfoque na Imagem
    img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)

    # Grava o pre-processamento para o OCR
    cv2.imwrite("output/roi-ocr.png", img_desfoque)

    # cv2.imshow("ROI", img_desfoque)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_desfoque


def ocrImageRoiPlaca(source):
    image = cv2.imread(source)

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    saida = pytesseract.image_to_string(image, lang='eng', config=config)

    return saida.strip()


if __name__ == "__main__":
    # encontrarRoiPlaca("placa_4.jpg")
    #pre = preProcessamentoRoiPlaca("output/roi29.png")
    ocr = ocrImageRoiPlaca("output/roi-ocr.png")
    print(F"Placa: {ocr}")

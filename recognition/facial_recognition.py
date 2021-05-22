import cv2 as cv
import os
from PIL import Image
import numpy as np
from django.contrib.admin.templatetags.admin_list import results

from recognition.exceptions import RecognitionError
from finder.settings import CLASSIFICATORS_ROOT


class Recognizer:
    def __init__(self):
        self.classificador_lbph = cv.face.LBPHFaceRecognizer_create()

    '''
    def preparar_imagem(self, diretorio):
        img = Image.open(diretorio).convert('L')
        img = np.asarray(img, 'uint8')
        face = self.extrair_regiao_interesse(img)
        if len(face) > 0:
            cv.imwrite('./img/ROI.jpg', face)
        else:
            raise Exception('Face não detectada.')
    '''

    def preparar_imagem(self, file):
        file = file.convert('L')
        # file.show()
        img = np.asarray(file, 'uint8')
        face = self.extrair_regiao_interesse(img)
        if len(face) > 0:
            Image.fromarray(face).show()
            return Image.fromarray(face)
        else:
            raise RecognitionError('Face não detectada.')

    def extrair_regiao_interesse(self, imagem):
        print(CLASSIFICATORS_ROOT + '/haarcascade_frontalface.xml')
        classificador_faces = cv.CascadeClassifier(CLASSIFICATORS_ROOT + '/haarcascade_frontalface.xml')
        classificador_olhos = cv.CascadeClassifier(CLASSIFICATORS_ROOT + '/haarcascade_eye.xml')
        faces = classificador_faces.detectMultiScale(imagem, scaleFactor=1.2, minNeighbors=5)
        if len(faces) > 0:
            x = faces[0][0]
            y = faces[0][1]
            lar = faces[0][2]
            alt = faces[0][3]
            face = imagem[y:y + lar, x:x + alt]
            olhos = classificador_olhos.detectMultiScale(face, scaleFactor=1.2, minNeighbors=5)
            if len(olhos) >= 1:
                face = cv.resize(face, (200, 200))
                return face
        return []

    def read_classifier_rec(self):
        self.classificador_lbph.read('classificadorLBPH.yml')

    def preparar_treinamento(self, diretorio_entrada, diretorio_saida):
        for f in os.listdir(diretorio_entrada):
            if os.path.isfile(os.path.join(diretorio_entrada, f)):
                img = Image.open(os.path.join(diretorio_entrada, f)).convert('L')
                img = np.asarray(img, 'uint8')
                # img = cv.imread(os.path.join(diretorio_entrada, f), cv.IMREAD_GRAYSCALE)
                face = self.extrair_regiao_interesse(img)
                if len(face) > 0:
                    cv.imwrite(diretorio_saida + '/' + f, face)

    def coletar_treinamento(self, diretorio_treinamento):
        dados_img, sujeitos = [], []
        for f in os.listdir(diretorio_treinamento):
            if os.path.isfile(os.path.join(diretorio_treinamento, f)):
                img = Image.open(os.path.join(diretorio_treinamento, f)).convert('L')
                img = np.asarray(img, 'uint8')
                # img = cv.imread(os.path.join(diretorio_treinamento, f), cv.IMREAD_GRAYSCALE)
                dados_img.append(img)
                sujeitos.append(int(f[1:3]))
        dados_img = np.asarray(dados_img, dtype=np.int32)
        sujeitos = np.asarray(sujeitos, dtype=np.int32)
        return dados_img, sujeitos

    def treinar_LBPH(self, diretorio_imagens):
        '''
        yale - 3,8,6,6 - 73.33

        '''
        lbph = cv.face.LBPHFaceRecognizer_create()
        dados_img, sujeitos = self.coletar_treinamento(diretorio_imagens)
        lbph.train(dados_img, sujeitos)
        lbph.write('classificadorLBPH.yml')

    def train_lbph(self, images, labels):
        lbph = cv.face.LBPHFaceRecognizer_create(neighbors=16, radius=2)
        lbph.train(images, labels)
        lbph.write('classificadorLBPH.yml')

    def reconhecer_lbph(self, img):
        '''Reconhece a face em uma imagem já preparada e com suas dimensões já recortadas'''
        # lbph = cv.face.LBPHFaceRecognizer_create()
        # lbph.read('classificadorLBPH.yml')
        return self.classificador_lbph.predict(img)

    def preparar_reconhecer_lbph(self, img):
        '''Extraxi face, e realzia o reocnhecimento'''
        img = self.preparar_imagem(img)
        self.read_classifier_rec()
        return self.classificador_lbph.predict(np.asarray(img, 'uint8'))

    def treinar_fiserface(self, diretorio_imagens):  # 5
        fischer = cv.face.FisherFaceRecognizer_create()
        dados_img, sujeitos = self.coletar_treinamento(diretorio_imagens)
        fischer.train(dados_img, sujeitos)
        fischer.write('classificadorffaces.yml')

    def reconhecer_fischer(self, img):
        fischer = cv.face.FisherFaceRecognizer_create()
        fischer.read('classificadorffaces.yml')
        return fischer.predict(img)

    def testar_acertos(self, diretorio_treinamento, nomecsv_saida, qtd_fotos):
        acertos = 0
        erros = 0
        imagens = 0
        for f in os.listdir(diretorio_treinamento):
            print(imagens, f)
            if os.path.isfile(os.path.join(diretorio_treinamento, f)):
                img = Image.open(os.path.join(diretorio_treinamento, f)).convert('L')
                img = np.asarray(img, 'uint8')
                # img = cv.imread(os.path.join(diretorio_treinamento, f), cv.IMREAD_GRAYSCALE)
                sujeito = int(f[1:3])
                sujeito_predicao, con = self.reconhecer_lbph(img)

                if sujeito == sujeito_predicao:
                    acertos += 1
                else:
                    erros += 1
                imagens += 1


class FacialRecognition:
    def __init__(self, classifier_dir="classificadorLBPH.yml"):
        self.classifier = cv.face.LBPHFaceRecognizer_create()
        self.classifier_dir = classifier_dir

    def resize_image(self, image_pil, width, height):
        '''
        Resize PIL image keeping ratio and using white background.
        '''
        ratio_w = width / image_pil.width
        ratio_h = height / image_pil.height
        if ratio_w < ratio_h:
            # It must be fixed by width
            resize_width = width
            resize_height = round(ratio_w * image_pil.height)
        else:
            # Fixed by height
            resize_width = round(ratio_h * image_pil.width)
            resize_height = height
        image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
        background.paste(image_resize, offset)
        return background.convert('RGB')


    def prepare_image(self, file, ignore_eyes=False):
        file = file.convert('L')
        #file.show()
        img = np.asarray(file, 'uint8')
        face = self.extract_roi_face(img, ignore_eyes)
        if len(face) > 0:
            return Image.fromarray(face)
        else:
            raise RecognitionError('Face não detectada.')

    def extract_roi_face(self, img, ignore_eyes=False):
        faces_classifier = cv.CascadeClassifier(CLASSIFICATORS_ROOT + '/haarcascade_frontalface.xml')
        faces = faces_classifier.detectMultiScale(img)
        if len(faces) > 0:
            x = faces[0][0]
            y = faces[0][1]
            lar = faces[0][2]
            alt = faces[0][3]
            face = img[y:y + lar, x:x + alt]
            face = cv.resize(face, (200, 200), interpolation=cv.INTER_LANCZOS4)
            if not ignore_eyes:
                if len(self.detect_eyes(face)) >= 2:
                    return face
                else:
                    return []
            else:
                return face
        return []

    def detect_eyes(self, img):
        eyes_classifier = cv.CascadeClassifier(CLASSIFICATORS_ROOT + '/haarcascade_eye.xml')
        eyes = eyes_classifier.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        if len(eyes) >= 2:
            return eyes
        return []

    def train_lbph(self, images, labels):
        lbph = cv.face.LBPHFaceRecognizer_create()
        lbph.train(images, labels)
        os.remove(self.classifier_dir)
        lbph.write(self.classifier_dir)

    def recognize_lbph(self, img):
        '''Reconhece a face em uma imagem já preparada e com suas dimensões já recortadas'''
        self.classifier.read(self.classifier_dir)
        return self.classifier.predict(img)

    def recognize_raw_img(self, img, ignore_eyes=False):
        '''Extraxi face, e realzia o reocnhecimento'''
        img = self.prepare_image(img, ignore_eyes)
        return self.recognize_lbph(np.asarray(img, 'uint8'))

    def webcam_capture(self):
        cam = cv.VideoCapture(0)
        font = cv.FONT_HERSHEY_COMPLEX_SMALL
        self.classifier.read(self.classifier_dir)
        while True:
            connected, img = cam.read()
            width, height = 200, 200
            grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces_classifier = cv.CascadeClassifier(CLASSIFICATORS_ROOT + '/haarcascade_frontalface.xml')
            detected_faces = faces_classifier.detectMultiScale(grey_image)
            for (x, y, w, h) in detected_faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                img_face = cv.resize(grey_image[y:y + h, x:x + w], (width, height))
                id, confidence = self.classifier.predict(cv.flip(img_face, 1))
                cv.putText(img, str(id) + ' - ' + str(confidence), (x, y + (h+35)), font, 2, (0, 0, 255))

            cv.imshow("Face", img)
            if cv.waitKey(1) == ord('q'):
                break
        cam.release()
        cv.destroyAllWindows()

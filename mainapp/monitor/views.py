from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
import cloudinary.uploader
from .apps import *
import numpy as np
import urllib
import cv2


# Create your views here.
class UploadView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    # MutliPartParser : analyse le contenu du formulaire HTML en plusieurs parties, qui prend en charge les téléchargements de fichiers
    # JSONPArser : analyse JSON le contenu de la requête. request.datasera alimenté par un dictionnaire de données.
    @staticmethod
    def post(request):

        def image_processing(img_url, shape_image):
            ## TRAITEMENT DE L'IMAGE ##
            req = urllib.request.urlopen(img_url)
            # Ouvrez l'url de l'image intégré au compte Cloudinary
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            # Convertir l'entrée en tableau, dtype : type de données unsigned integer 8 bits
            image = cv2.imdecode(arr, -1)
            # imdecode est utilisée pour lire des données d'image et les convertir en format d'image.
            # Cette méthode est généralement utilisée pour charger efficacement l'image à partir d'Internet.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convertir une image d'un espace colorimétrique à un autre : BGR -> vers RGB
            image = cv2.resize(image, (shape_image))
            # Redimensionner l'image en 224 x 224 ou 299 x 299
            image = np.array(image) / 255
            # Normalisation en intervalle de [0,1]
            image = np.expand_dims(image, axis=0)
            # Développer la forme d'un tableau en ajoutant une nouvelle dimensionnalité au tableau
            return image

        ## RECUPERER L'IMAGE ET ENREGISTREMENT SOUS CLOUDINARY ##
        file = request.data.get('picture')
        # Requête HTTP de type get pour récupérer l'image envoyé par l'utilisateur
        upload_data = cloudinary.uploader.upload(file)
        # La méthode Cloudinary upload effectue un appel d'API de téléchargement authentifié par HTTPS
        # pour télécharger des actifs dans le cloud de cloudinary
        # print(upload_data)
        img_url = upload_data['url']
        # Récupérer l'url de l'image présente dorénavant dans un espace Cloudinary

        ## CHARGEMENT DES MODELES ##
        resnet_tumor = ResNetModelConfig.model
        vgg_tumor = VGGModelConfig.model
        inception_resnet_v2_tumor = InceptionResnetV2ModelConfig.model

        ## PREDICTION : MODELE VGG ##
        image = image_processing(img_url, (224, 224))
        vgg_pred = vgg_tumor.predict(image)
        # Effectuer une prédiction avec le modèle vgg
        probability = vgg_pred[0]
        print("probability vgg :")
        print(probability)
        if probability[0] > 0.5:
            vgg_tumor_pred = str('%.2f' % (probability[0] * 100) + '% Tumeur non détectée')
        else:
            vgg_tumor_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% Tumeur détectée')

        ## PREDICTION : MODELE RESNET ##
        image = image_processing(img_url, (224, 224))
        resnet_pred = resnet_tumor.predict(image)
        # Effectuer une prédiction avec le modèle resnet
        probability = resnet_pred[0]
        print("probability resnet :")
        print(probability)
        if probability[0] > 0.5:
            resnet_tumor_pred = str('%.2f' % (probability[0] * 100) + '% Tumeur non détecté')
        else:
            resnet_tumor_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% Tumeur détectée')

        return Response({
            # La réponse rendue au client :
            'status': 'success',
            'data': upload_data,
            'url': img_url,
            'vgg_tumor_pred': vgg_tumor_pred,
            'resnet_tumor_pred': resnet_tumor_pred,
        }, status=201)

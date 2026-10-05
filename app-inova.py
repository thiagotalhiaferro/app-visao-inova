import os
import sys

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def analisar_imagem(caminho_imagem):
    endpoint = os.getenv("VISION_ENDPOINT")
    chave = os.getenv("VISION_KEY")

    if not endpoint or not chave:
        print("ERRO: as variáveis VISION_ENDPOINT e VISION_KEY não estão configuradas.")
        sys.exit(1)

    if not os.path.isfile(caminho_imagem):
        print(f"ERRO: a imagem '{caminho_imagem}' não foi encontrada.")
        sys.exit(1)

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(chave)
    )

    with open(caminho_imagem, "rb") as arquivo:
        imagem = arquivo.read()

    print("\n" + "=" * 60)
    print("       INOVA TRÔNICA - VISÃO COMPUTACIONAL")
    print("=" * 60)

    print("\nImagem selecionada:")
    print(f"  {caminho_imagem}")

    print("\nEnviando imagem para o Azure AI Vision...")

    resultado = client.analyze(
        image_data=imagem,
        visual_features=[
            VisualFeatures.CAPTION
        ],
        gender_neutral_caption=True
    )

    print("\nResultado do Serviço Cognitivo:")

    if resultado.caption:
        print("\nDescrição da imagem:")
        print(f"  {resultado.caption.text}")

        print("\nConfiança da análise:")
        print(f"  {resultado.caption.confidence:.2%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    caminho = input("Digite o caminho da imagem: ").strip()

    try:
        analisar_imagem(caminho)
    except Exception as erro:
        print("\nERRO AO CONSULTAR O AZURE VISION:")
        print(erro)

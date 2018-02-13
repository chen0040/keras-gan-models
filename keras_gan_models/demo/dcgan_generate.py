from keras_gan_models.library.dcgan import DCGan
from keras_gan_models.library.utility.image_loader import load_and_scale_images


def main():
    model_dir_path = './models'

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image = gan.generate_image()
        image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '.png')

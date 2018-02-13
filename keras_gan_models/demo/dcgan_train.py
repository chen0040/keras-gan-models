from keras_gan_models.library.dcgan import DCGan
from keras_gan_models.library.utility.image_loader import load_and_scale_images


def main():
    image_dir_path = './data/images'
    model_dir_path = './models'

    img_width = 64
    img_height = 64
    img_channels = 3

    gan = DCGan()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 200

    images = load_and_scale_images(image_dir_path, '.png', img_width, img_height)

    batch_size = 4
    gan.fit(model_dir_path, images=images, batch_size=batch_size)


if __name__ == '__main__':
    main()


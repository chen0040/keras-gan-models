from keras_gan_models.library.dcgan import DCGan


def main():
    model_dir_path = './models'

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image = gan.generate_image()
        image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '.png')


if __name__ == '__main__':
    main()


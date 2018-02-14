from keras_gan_models.library.dcgan import DCGan


def main():
    model_dir_path = './models'

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image = gan.generate_image()
        img_path = './data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '.png'
        print('generating: ', img_path)
        image.save(img_path)


if __name__ == '__main__':
    main()


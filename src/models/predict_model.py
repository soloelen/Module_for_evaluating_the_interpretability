import torch
from matplotlib import pyplot as plt

from ..domain.labels import Label
from ..data.data_transform import transform
from ..config.config import DEVICE


def get_10_classes_sample(original_test_data):
    """Получение семпла из 10 различных тестовых изображений"""
    num_images_get = 0
    ten_images_CIFAR = {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        9: None
    }

    for image, label in original_test_data:
        if num_images_get >= 10:
            break
        if ten_images_CIFAR[label] is None:
            ten_images_CIFAR[label] = image
            num_images_get += 1
    return ten_images_CIFAR


def predict_10_classes(model, original_test_data):
    """Предсказание классов для тестового семпла"""
    ten_images_CIFAR = get_10_classes_sample(original_test_data)
    ret = {
        "original_images": [],
        "transformed_images": [],
        "predicted_labels": [],
        "real_labels": []
    }

    correct = 0
    model.eval()
    with torch.no_grad():
        for real_label, image in ten_images_CIFAR.items():
            transformed_image = transform['test'](image).to(DEVICE)
            output = model(transformed_image.unsqueeze(0))
            output = torch.argmax(output, dim=1)
            predicted_label_name = Label(int(output[0])).name
            real_label_name = Label(real_label).name
            if predicted_label_name == real_label_name:
                correct += 1
            ret['original_images'].append(image)
            ret['transformed_images'].append(transformed_image)
            ret['predicted_labels'].append(predicted_label_name)
            ret['real_labels'].append(real_label_name)

    return ret


def show_predictions(original_images, predicted_labels, real_labels):
    """Визуализация предсказаний совместно с изображениями"""
    for image, pred, real in zip(original_images, predicted_labels, real_labels):
        plt.imshow(image)
        plt.title(f"{real}(real)  |  {pred}(pred)")
        plt.show()

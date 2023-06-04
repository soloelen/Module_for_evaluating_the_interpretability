from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .models.train_model import train, show_metrics
from .models.predict_model import predict_10_classes, show_predictions
from .models.models import ResNet18, VGG16
from .data.make_dataset import get_dataloaders
from .config.config import PATH_TO_MODEL


def train_model_pipeline(model_name, freeze_conv=True, filename=None):
    """
    Пайплайн обучения указанной модели.
    :param model_name: ["resnet" or "vgg"]
    :param freeze_conv: Заморозить верхние слои или нет
    :param filename: Имя файла сохраняемой модели.
    """
    if model_name == "resnet":
        model = ResNet18(freeze_conv=freeze_conv)
    elif model_name == "vgg":
        model = VGG16(freeze_conv=freeze_conv)
    else:
        raise NameError

    criterion = nn.CrossEntropyLoss()
    if model_name == "vgg":
        param_groups = [
            {'params': model.model.features.parameters(), 'lr': 0.0001},
            {'params': model.model.classifier.parameters(), 'lr': 0.001}
        ]
    else:
        param_groups = [

            {'params': model.model.conv1.parameters(), 'lr': 0.0001},
            {'params': model.model.layer1.parameters(), 'lr': 0.0001},
            {'params': model.model.layer2.parameters(), 'lr': 0.0001},
            {'params': model.model.layer3.parameters(), 'lr': 0.0001},
            {'params': model.model.layer4.parameters(), 'lr': 0.0001},
            {'params': model.model.fc.parameters(), 'lr': 0.001}
        ]
    optimizer = Adam(param_groups)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    train_loader, val_loader, original_test_data = get_dataloaders(get_original_test_data=True)
    history = train(model.model, train_loader, val_loader, optimizer, criterion, lr_scheduler)

    # Evaluation
    show_metrics(history, f"{model_name} metrics")
    pred = predict_10_classes(model=model.model, original_test_data=original_test_data)
    show_predictions(original_images=pred['original_images'],
                     predicted_labels=pred['predicted_labels'],
                     real_labels=pred['real_labels'])
    # Saving
    if filename is None:
        filename = f"{model_name}.pth"
    model.save(PATH_TO_MODEL.format(filename))


import torchvision


class Model():
    """The ShuffleNet model."""
    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type.startswith('shufflenet')
                and len(model_type.split('_')) == 2
                and float(model_type.split('_')[1]) in [0.5, 1.0, 1.5, 2.0])

    @staticmethod
    def get_model(model_type, num_classes=10, pretrained=False):
        if not Model.is_valid_model_type(model_type):
            raise ValueError(
                'Invalid ShuffleNet model type: {}'.format(model_type))

        if model_type == 'shufflenet_0.5':
            return torchvision.models.shufflenet_v2_x0_5(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'shufflenet_1.0':
            return torchvision.models.shufflenet_v2_x1_0(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'shufflenet_1.5':
            return torchvision.models.shufflenet_v2_x1_5(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'shufflenet_2.0':
            return torchvision.models.shufflenet_v2_x2_0(
                pretrained=pretrained, num_classes=num_classes
            )

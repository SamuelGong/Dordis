import torchvision


class Model():
    """The ShuffleNet model."""
    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type.startswith('efficientnet')
                and len(model_type.split('_')) == 2
                and model_type.split('_')[1] in [
                    'b0', 'b1', 'b2', 'b3',
                    'b4', 'b5', 'b6', 'b7',
                ])

    @staticmethod
    def get_model(model_type, num_classes=10, pretrained=False):
        if not Model.is_valid_model_type(model_type):
            raise ValueError(
                'Invalid EffcientNet model type: {}'.format(model_type))

        if model_type == 'efficientnet_b0':
            return torchvision.models.efficientnet_b0(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b1':
            return torchvision.models.efficientnet_b1(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b2':
            return torchvision.models.efficientnet_b2(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b3':
            return torchvision.models.efficientnet_b3(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b4':
            return torchvision.models.efficientnet_b4(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b5':
            return torchvision.models.efficientnet_b5(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b6':
            return torchvision.models.efficientnet_b6(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model_type == 'efficientnet_b7':
            return torchvision.models.efficientnet_b7(
                pretrained=pretrained, num_classes=num_classes
            )

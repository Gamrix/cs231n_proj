import torch


class L2Loss(torch.nn.Module):
    def forward(self, y_pred, y_true):
        diffsq = (y_pred - y_true) **2
        return torch.mean(torch.sum(diffsq.view((-1, 224*224*3)), dim=1))


class TextureLoss(torch.nn.Module):
    """
    Texture Loss is a L2 loss that also penalizes for deltas (textures) over various distances.
    """
    def __init__(self, texture_loss_weight=2):
        self.texture_loss_weight = texture_loss_weight
        super(TextureLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = self.l2_loss(y_pred, y_true)
        text_loss = 0
        for i in range(3):
            dist = 2 ** 3
            text_loss += self.l2_loss(self.delta_x(y_pred, dist), self.delta_x(y_true, dist))
            text_loss += self.l2_loss(self.delta_y(y_pred, dist), self.delta_y(y_true, dist))

        return loss + self.texture_loss_weight * text_loss

    @staticmethod
    def delta_x(image, offset):
        return image[:, :, :, :-offset] - image[:, :, :, offset:]

    @staticmethod
    def delta_y(image, offset):
        return image[:, :, :-offset, :] - image[:, :, offset:, :]

    @staticmethod
    def l2_loss(y_pred, y_true):
        N = y_pred.size()[0]
        diffsq = (y_pred - y_true) **2
        return torch.mean(torch.sum(diffsq.view((N, -1)), dim=1))

class TextureLoss2(torch.nn.Module):
    """
    Texture Loss is a L2 loss that also penalizes for deltas (textures) over various distances.
    """
    def __init__(self, texture_loss_weight=2):
        self.texture_loss_weight = texture_loss_weight
        super(TextureLoss2, self).__init__()

    def forward(self, y_pred, y_true):
        loss = self.l1_loss(y_pred, y_true)
        text_loss = 0
        for i in range(1):
            dist = 2 ** 3
            text_loss += self.l2_loss(torch.abs(self.delta_x(y_pred, dist)), torch.abs(self.delta_x(y_true, dist)))
            text_loss += self.l2_loss(torch.abs(self.delta_y(y_pred, dist)), torch.abs(self.delta_y(y_true, dist)))

        return loss + self.texture_loss_weight * text_loss

    @staticmethod
    def delta_x(image, offset):
        return image[:, :, :, :-offset] - image[:, :, :, offset:]

    @staticmethod
    def delta_y(image, offset):
        return image[:, :, :-offset, :] - image[:, :, offset:, :]

    @staticmethod
    def l2_loss(y_pred, y_true):
        N = y_pred.size()[0]
        diffsq = (y_pred - y_true) **2
        return torch.mean(torch.sum(diffsq.view((N, -1)), dim=1))

    @staticmethod
    def l1_loss(y_pred, y_true):
        N = y_pred.size()[0]
        diff = torch.abs(y_pred - y_true)
        return torch.mean(torch.sum(diff.view((N, -1)), dim=1))

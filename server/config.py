
import torch
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_VAL_SPLIT = [0.8, 0.2]

IMAGE_TRANFORM_TRAINING = transforms.Compose([transforms.RandomResizedCrop(200),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])

IMAGE_TRANFORM_INFERENCE = transforms.Compose([transforms.Resize(200),
                                               transforms.CenterCrop(200),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])

BATCH_SIZE = 32

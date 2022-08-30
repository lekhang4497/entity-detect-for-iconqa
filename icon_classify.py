import torch
from torchvision import models
from PIL import Image, ImageOps
from torchvision import transforms
import glob

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Load Icon classifier components')

# Load Resnet model
model_path = "/home/khangln/JAIST_DRIVE/WORK/IconQA/saved_models/icon_classification_ckpt/icon_resnet101_LDAM_DRW_lr0.01_0/ckpt.epoch66_best.pth.tar"
print("loading pretrained models on icon data: ", model_path)
checkpoint = torch.load(model_path, map_location='cuda')

model = models.__dict__['resnet101'](
    pretrained=False, num_classes=377)  # icon classes from Icon645
model.load_state_dict(checkpoint['state_dict'])
model = model.eval().to(device)

# Load id2name mapping
names = []
for t in glob.glob('/home/khangln/JAIST_DRIVE/WORK/IconQA/icon_data/colored_icons_final/*'):
    names.append(t.split('/')[-1])

names.sort()
assert len(names) == 377
id2name = {i: name for i, name in enumerate(names)}

print('Finish loading Icon classifier components')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


def crop_margin(img_fileobj):
    ivt_image = ImageOps.invert(img_fileobj)
    bbox = ivt_image.getbbox()  # [left, top, right, bottom]
    cropped_image = img_fileobj.crop(bbox)

    return cropped_image


def add_padding(img, padding=2):
    """Add borders to the 4 sides of an image"""
    desired_size = max(img.size) + padding * 2

    if img.size[0] < desired_size or img.size[1] < desired_size:
        delta_w = desired_size - img.size[0]
        delta_h = desired_size - img.size[1]
        padding = (max(delta_w // 2, 0), max(delta_h // 2, 0),
                   max(delta_w - (delta_w // 2), 0), max(delta_h - (delta_h // 2), 0))
        img = ImageOps.expand(img, padding, (255, 255, 255))
    return img


def classify(img_file):
    global id2name
    img = Image.open(img_file)
    img = img.convert('RGB')

    # We need crop and pad the choice images for icon_pretrained mode
    # Because the icon_pretrained model is pretrained on our icon data
    img = crop_margin(img)
    img = add_padding(img)
    img = transform(img)

    img = img.to(device)

    img = img.unsqueeze(0)

    with torch.no_grad():
        result = model(img)
        class_id = torch.argmax(result.squeeze()).item()
        return id2name[class_id]


def classify_img(img):
    global id2name
    # img = Image.open(img_file)
    # img = img.convert('RGB')

    # We need crop and pad the choice images for icon_pretrained mode
    # Because the icon_pretrained model is pretrained on our icon data
    img = crop_margin(img)
    img = add_padding(img)
    img = transform(img)

    img = img.to(device)

    img = img.unsqueeze(0)

    with torch.no_grad():
        result = model(img)
        class_id = torch.argmax(result.squeeze()).item()
        return id2name[class_id]


# names = []
# for t in glob.glob('/home/khangln/JAIST_DRIVE/WORK/IconQA/icon_data/colored_icons_final/*'):
#     names.append(t.split('/')[-1])

# names.sort()
# assert len(names) == 377
# id2name = {i: name for i, name in enumerate(names)}


# for t in glob.glob('/home/khangln/JAIST_DRIVE/WORK/IconQA/icon_data/colored_icons_final/ball/*.png'):
#     i = classify(t)
#     print(id2name[i])

# i = classify(
#     '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/test/choose_img/105756/choice_0.png')
# print(id2name[i])

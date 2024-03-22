from utils.plots import Annotator, colors
from yolo9_infer import obj_detect
from parseq_infer import init_model, recognize
import torch
from PIL import Image
import cv2

# Path to the model of Yolov9 and parseq-tiny
yolo_weights = r"./pretrained_model/yolov9-c-converted-best-5class.pt"
parseq_model = r"./pretrained_model/parseq-tiny-epoch=7-step=298-val_accuracy=99.0909-val_NED=99.0909.pt"
# path to the image you want to test
image_source = r"./test_set/01.jpg"
device = "cpu"

# First, we detect using YOLOv9
det = obj_detect(weights=yolo_weights, source=image_source, device=device)


img = Image.open(image_source).convert('RGB')
im0 = img.copy()
# annotator used for visualize results
annotator = Annotator(im0, line_width=3, pil=True)
# Next, we start recognize using parseq
parseq = init_model(checkpoint=parseq_model, device=device)

for *xyxy, conf, cls in reversed(det):
    xy1xy2 = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
    print(xy1xy2)
    c = int(cls)  # integer class
    if c == 2 or c == 3:  # change this to "c == 0 or c == 1" if run Yolov9 detect 3 class
        im1 = img.crop(tuple(xy1xy2))
        with torch.no_grad():
            label = recognize(parseq,img=im1,device=device)

        # From here, I change the recognition result to truncate not use information from the result
        # For example, "A." will be "A", "Question18" will be "18"
        # But for testing and compute accuracy of 2e2 model, we do not need this part of code
        if c == 3:  # change this to "c == 1" if run Yolov9 detect 3 class
            if len(label) > 0:
                label = label[0]
        if c == 2:  # change this to "c == 0" if run Yolov9 detect 3 class
            new_label = ""
            for char in label:
                if char.isdigit():
                    new_label += char
            label = new_label
        # End truncating here
        # Next, I draw box and recognized text on the image for visualization
        annotator.box_label(xyxy, label, color=colors(c, True))
img = annotator.result()
save_path = r"./test_set/result.jpg"
cv2.imwrite(save_path, img)

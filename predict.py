import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from gradcam import save_overlay
from gwo import Hyperparameters
from models import Classifier


def main():
    parser = argparse.ArgumentParser(description='Predict leaf disease from a trained checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--gradcam', type=Path, help='Save overlay; must not already exist.')
    args = parser.parse_args()
    if args.gradcam and args.gradcam.exists():
        parser.error('Grad-CAM file already exists; choose another output.')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Classifier(len(checkpoint['classes']), Hyperparameters(**checkpoint['config']),
                       checkpoint['architecture'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transform = transforms.Compose([transforms.Resize((checkpoint['image_size'],) * 2),
                                    transforms.ToTensor(), transforms.Normalize(
                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with Image.open(args.image) as original:
        image = transform(original.convert('RGB')).unsqueeze(0)
        with torch.inference_mode():
            scores = model(image).softmax(1)[0]
        ranking = torch.argsort(scores, descending=True)[:min(3, len(scores))]
        output = [{'class': checkpoint['classes'][i], 'score': scores[i].item()} for i in ranking.tolist()]
        if args.gradcam:
            save_overlay(model, image, original, args.gradcam)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

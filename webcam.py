#coding:utf-8
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import numpy as np

def argmax(prediction):

    classes = ('paper', 'rock', 'scissors')

    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = classes[prediction]

    return result,score

def main():

    INPUT_SIZE = (224, 224)

    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    inference_transformation = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    device = torch.device('cuda:0')

    net = models.resnet101(num_classes=3)
    net.load_state_dict(torch.load('resnet101-rock-scissors-paper.pth'))
    net.to(device)
    net.eval()

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess frame
        img = Image.fromarray(frame)
        img_tensor = inference_transformation(img).unsqueeze_(0)
        img_tensor = img_tensor.to(device)
        # Prediction step
        output = net(Variable(img_tensor))
        result, score = argmax(output)


        if float(score) >= 0.55:

            prediction = result
            print('Prediction : %s' % prediction)
        else:
            prediction = "Nothing"
            print('No signal')


        if prediction =='rock':
            a = 'status/paper.jpg'
        elif prediction =='scissors':
            a = 'status/rock.jpg'
        elif prediction == 'paper':
            a = 'status/scissors.jpg'
        else:
            a = 'no.jpg'
        ans = cv2.imread(a)
        ans = cv2.resize(ans,(640,480))
        cv2.imshow('aaa',ans)
        cv2.putText(img=frame, text=prediction, org=(70, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.,
                    color=(255, 255, 255))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()

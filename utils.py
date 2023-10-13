import cv2
import numpy as np
from torchvision import transforms
from IPython.display import clear_output, display
from PIL import Image
import config
from Generator import UNetResNet18




transform = transforms.Compose([
            transforms.ToTensor()
        ])


new_transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE),  Image.BICUBIC),
            ])


def generate_image(generator,img,size):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(size,size))
    img = img.astype('float32')/255.0

    input_tensor = transform(img)
    input_tensor = input_tensor.to(config.DEVICE)
    input_tensor = input_tensor.unsqueeze(0)

    gen_output = generator(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).numpy()
    gen_output = np.clip(gen_output,0,1)
    img = np.expand_dims(img,axis=2)

    lab_img = np.concatenate([img,gen_output],axis=2)
    bgr = cv2.cvtColor(np.uint8(lab_img*255),cv2.COLOR_LAB2BGR)

    return bgr


def generate_images(generator, input_path):
    colored = cv2.imread(input_path)
    colored = cv2.cvtColor(colored,cv2.COLOR_BGR2RGB)
    img = cv2.imread(input_path,0)
    h,w = img.shape
    
    img = cv2.resize(img,(config.IMAGE_SIZE,config.IMAGE_SIZE))
    colored = cv2.resize(colored,(config.IMAGE_SIZE,config.IMAGE_SIZE))
    img = img.astype('float32')/255.0

    input_tensor = transform(img)
    input_tensor = input_tensor.to(config.DEVICE)
    input_tensor = input_tensor.unsqueeze(0)



    gen_output = generator(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = np.expand_dims(img,axis=2)

    lab_img = np.concatenate([img,gen_output],axis=2)
    bgr = cv2.cvtColor(np.uint8(lab_img*255),cv2.COLOR_LAB2RGB)

    # hsv_image = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    # hsv_image[:, :, 1] = hsv_image[:, :, 1] * 1.5 
    # bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    img = np.uint8(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)*255)
    
    final = np.concatenate((img,colored,bgr), axis=1)

    #Display the combined image
    clear_output(wait=True)
    display(Image.fromarray(final))
    # cv2.imshow('fin',final)
    # cv2.waitKey(0)

if __name__ == "__main__":
    gen = UNetResNet18(1,2).to(config.DEVICE)
    # gen.load_state_dict(torch.load('gen.pt'))
    generate_images(gen,'coco/0.jpg')
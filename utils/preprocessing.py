import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms


# augmentation CV2
class RandomClahe():
    def __init__(self, clipLimit=3, tileGridSize=(3,3), random_prob=0.5):
        self.base_clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.random_prob = random_prob

    def __call__(self, sample):
        img = sample['image']
        
        current_clipLimit = self.base_clipLimit
        if np.random.uniform() < self.random_prob:
            current_clipLimit += 3 * (np.random.uniform() - 0.5)
            
        clahe = cv2.createCLAHE(clipLimit=current_clipLimit, tileGridSize=self.tileGridSize)
            
        if len(img.shape) < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        
        img_copy = img.copy() 
        img_copy[...,0] = clahe.apply(img_copy[...,0])
        img_copy[...,1] = clahe.apply(img_copy[...,1])
        img_copy[...,2] = clahe.apply(img_copy[...,2])

        sample['image'] = img_copy
        
        return sample
    
    

    
# augmentation torch transforms
class RandomBright():
    def __init__(self, 
                 bright=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        self.base_bright = bright
        self.random_range = random_range
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img = sample['image']
            
            bright_delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_bright = self.base_bright + bright_delta
            
            img = transforms.functional.adjust_brightness(img, current_bright)

            sample['image'] = img
        
        return sample
    
    
    
    
class RandomContrast():
    def __init__(self, 
                 contrast=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        self.base_contrast = contrast
        self.random_range = random_range
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img = sample['image']
            
            delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_contrast = self.base_contrast + delta
            
            img = transforms.functional.adjust_contrast(img, current_contrast)

            sample['image'] = img
            
        return sample
    
    
    
    
class RandomGamma():
    def __init__(self, 
                 gamma=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        self.base_gamma = gamma
        self.random_range = random_range
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img = sample['image']
            
            delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_gamma = self.base_gamma + delta
            
            img = transforms.functional.adjust_gamma(img, current_gamma)

            sample['image'] = img
            
        return sample
    
    
    
    
class RandomHue():
    def __init__(self, 
                 hue=0,
                 random_range=0.1,
                 aug_prob=0.25):
        
        self.base_hue = hue
        self.random_range = random_range
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img = sample['image']
            
            delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_hue = self.base_hue + delta
            
            img = transforms.functional.adjust_hue(img, current_hue)

            sample['image'] = img
            
        return sample
    
    
    
    
class RandomSaturation():
    def __init__(self, 
                 saturation=1,
                 random_range=0.1,
                 aug_prob=0.25):
        
        self.base_saturation = saturation
        self.random_range = random_range
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img = sample['image']
            
            delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_saturation = self.base_saturation + delta
            
            img = transforms.functional.adjust_saturation(img, current_saturation)

            sample['image'] = img
            
        return sample


class RandomRotation():
    def __init__(self, 
                 angle=0,
                 aug_prob=0.25):
        
        self.max_angle = angle
        self.aug_prob = aug_prob
        
    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img, label = sample['image'], sample['label']
            
            delta = 2 * (np.random.uniform(0, self.max_angle) - self.max_angle / 2)
            current_angle = delta 
            
            img = transforms.functional.rotate(img, current_angle)
            label = transforms.functional.rotate(label, current_angle)

            sample['image'] = img
            sample['label'] = label
            
        return sample
    

class RandomNoise():
    def __init__(self,
                 noise=0,
                 random_range=0.1,
                 aug_prob=0.25):

        self.base_noise = noise
        self.random_range = random_range
        self.aug_prob = aug_prob

    def __call__(self, sample):
        if np.random.uniform() < self.aug_prob:
            img_pil = sample['image'] # Get the PIL Image
            img_np = np.array(img_pil)

            delta = 2 * (np.random.uniform(0, self.random_range) - self.random_range / 2)
            current_noise = np.abs(self.base_noise + delta)

            # Apply noise
            # Ensure img_np has a shape attribute when passed to np.random.normal
            noise_map = np.random.normal(0, current_noise, img_np.shape)
            img_np = img_np + noise_map

            # Normalize to [0, 255] and convert to uint8
            img_np = np.clip(img_np, 0, 255)
            img_np = img_np.astype(np.uint8)

            # Convert NumPy array back to PIL Image
            sample['image'] = Image.fromarray(img_np)

        return sample


class RandomBlack():
    def __init__(self,
                 dark_factor_range=(0.7, 0.9), 
                 contrast_factor_range=(1.0, 1.2),
                 aug_prob=0.25):


        self.dark_factor_range = dark_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.aug_prob = aug_prob

    def __call__(self, sample):

        if np.random.uniform() < self.aug_prob:
            img_pil = sample['image']


            dark_factor = np.random.uniform(self.dark_factor_range[0], self.dark_factor_range[1])
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(dark_factor)

            contrast_factor = np.random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(contrast_factor)

            sample['image'] = img_pil

        return sample


class RandomBlur():
    def __init__(self,
                 radius_range=(0.5, 2.0),
                 aug_prob=0.25):

        self.radius_range = radius_range
        self.aug_prob = aug_prob

    def __call__(self, sample):

        if np.random.uniform() < self.aug_prob:
            img_pil = sample['image']

            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

            sample['image'] = img_pil

        return sample


class ToPILImage():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img   = transforms.ToPILImage()(img)
        label = transforms.ToPILImage()(label)

        sample['image'] = img
        sample['label'] = label
        
        return sample
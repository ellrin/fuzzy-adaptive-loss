from torchvision import transforms

from utils.preprocessing import (RandomBlack, RandomBlur, RandomBright, RandomClahe,
                                 RandomContrast, RandomGamma, RandomHue, RandomNoise,
                                 RandomRotation, RandomSaturation, ToPILImage)

# ==== PATH ====
TRAIN_PATH         = './data/train/'
VALID_PATH         = './data/valid/'
TEST_PATH          = './data/test/'
QUALITY_MODEL_PATH = './weights/quality_model.pth'
SAVE_PATH          = './results/'

# ==== segmentation model ====
STRUCTURE     = 'unet'
BACKBONE      = 'efficientnet-b2'
PRE_TRAINED   = 'imagenet'
IMGSIZE       = 224

# ==== quality model ====
QMODEL        = 'efficientnet'
Q_LEVELS      = 5

# ==== training ====
DEVICE        = 'cuda:0'
SEED          = 4444444
BATCH         = 8
EPOCHS        = 50
LEARNING_RATE = 2e-4
TRANSFORM_RATE = 0.75
NUM_WORKERS   = 4

# ==== loss ====
LOSS_TYPE     = 'dice'      # dice, bce
WEIGHT_MODE   = 'choquet'   # choquet, linear, none
LINEAR_LAMBDA = 0.6         # linear baseline: w = lambda * q + (1 - lambda) * a

# ==== fuzzy: quality membership (Gaussian centers / shared spread) ====
Q_LOW_MU      = 0.3
Q_MEDIUM_MU   = 0.5
Q_HIGH_MU     = 0.7
Q_SIGMA       = 0.075

# ==== fuzzy: area ratio membership ====
A_SMALL_MU    = 0.005
A_MEDIUM_MU   = 0.024
A_LARGE_MU    = 0.043
A_SIGMA       = 0.01

# ==== fuzzy: Choquet consequent ====
ETA           = 0.6         # density mass allocated to the quality cue
SCALE         = 0.6         # overall density magnitude, g_q = SCALE * ETA, g_a = SCALE * (1 - ETA)
PROTOTYPES    = [0.001, 0.5, 1.0]   # negligible, moderate, strong semantic contribution
FUZZY_EPS     = 1e-6

# ==== augmentation ====
clahe_clip_limit     = 8
clahe_tile_grid_size = (7, 7)
clahe_aug_prob       = 0.75

bright_range         = 0.75
bright_aug_prob      = 0.5

contrast_range       = 0.6
contrast_aug_prob    = 0.5

gamma_range          = 0.645
gamma_aug_prob       = 0.5

hue_range            = 0.02
hue_aug_prob         = 0

sat_range            = 0.02
sat_aug_prob         = 0

noise_range          = 0.875
noise_aug_prob       = 0.5

black_dark_range     = (0.4, 1.4)
black_contrast_range = (1.0, 1.2)
black_aug_prob       = 0.5

blur_radius_range    = (0.1, 8.0)
blur_aug_prob        = 0.75

rotate_angle         = 45

AUGMENTATION = transforms.Compose([
    RandomClahe(clahe_clip_limit, clahe_tile_grid_size, random_prob=clahe_aug_prob),
    ToPILImage(),
    RandomBright(random_range=bright_range, aug_prob=bright_aug_prob),
    RandomContrast(random_range=contrast_range, aug_prob=contrast_aug_prob),
    RandomGamma(random_range=gamma_range, aug_prob=gamma_aug_prob),
    RandomHue(random_range=hue_range, aug_prob=hue_aug_prob),
    RandomSaturation(random_range=sat_range, aug_prob=sat_aug_prob),
    RandomNoise(random_range=noise_range, aug_prob=noise_aug_prob),
    RandomBlack(dark_factor_range=black_dark_range, contrast_factor_range=black_contrast_range, aug_prob=black_aug_prob),
    RandomBlur(radius_range=blur_radius_range, aug_prob=blur_aug_prob),
    RandomRotation(rotate_angle),
])

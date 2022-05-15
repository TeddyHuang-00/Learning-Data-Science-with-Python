# # Shared model parameters
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 50

# # Random seed for torch and numpy
SEED = 0

# # PGD attack
ATTACK_EPS = 16 / 255
ATTACK_ALPHA = 2 / 255
ATTACK_ITER = 4

# # Quantization
# Simulated quantization for quantization training
# -1 means do not simulate quantization during training (Post-train quantization)
# 1 means simulate quantization every step during training (Quantization-aware training)
# Any number >1 means simulate quantization occasionally during training (Semi-quantization-aware training)
# If to perform semi-QAT, 4 is empirically a good value
QUANT_WEIGHT = True
QUANT_THRESH = True
QUANT_WEIGHT_BITS = 8
QUANT_THRESH_BITS = 8

# # Regularization
REG_SPEC = 5e-1
REG_ORTH = 5e-2

# # Random self-ensemble
RSE_INIT = 2e-1
RSE_INNER = 1e-1
RSE_NUM = 5

import torch


checkpoint = torch.load("weights/model_trained.pth")

# Print keys in the checkpoint dictionary
print("Keys in the checkpoint:")
for key in checkpoint.keys():
    print(key)

# Access specific components in the checkpoint
model_state_dict = checkpoint["model_state"]
print("Model State Dictionary:")
print(model_state_dict.keys())
# print(model_state_dict)

# optimizer_state_dict = checkpoint['optimizer_state']
# print("Optimizer State Dictionary:")
# print(optimizer_state_dict.keys())
# print(optimizer_state_dict)
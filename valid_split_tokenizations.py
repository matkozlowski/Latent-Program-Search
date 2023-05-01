import random

VALID_SPLIT = 0.3

input_file = "tokenizations.txt"
valid_file = "valid_tokenizations.txt"
train_file = "train_tokenizations.txt"

# Read all lines from the input file
with open(input_file, "r") as file:
    lines = file.readlines()
    
# Shuffle the lines
random.shuffle(lines)

# Calculate the number of lines for the validation set
num_lines_valid = int(len(lines) * VALID_SPLIT)

lines_valid = lines[:num_lines_valid]
lines_train = lines[num_lines_valid:]

with open(valid_file, "w") as file:
    file.writelines(lines_valid)
    
with open(train_file, "w") as file:
    file.writelines(lines_train)
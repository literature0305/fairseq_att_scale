import torch
import numpy
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

input_file='errlog005-2_decode'
with open(input_file, 'r') as f:
    lines=f.readlines()

total_bleu1=torch.zeros(1000)
total_bleu2=torch.zeros(1000)
total_bleu3=torch.zeros(1000)
total_bleu4=torch.zeros(1000)
total_sample=torch.ones(1000)

max_length=120

for line in lines:
    if line[:2] == 'S-':
        source_sent=line.split('\t')[1]
    elif line[:2] == 'T-':
        reference=line.split('\t')[1]
    elif line[:2] == 'H-':
        candidate=line.split('\t')[2]
        hyp_score=line.split('\t')[1]

    elif line[:2] == 'D-':
        ref_length= len(reference.split())
        reference = [reference.split()]
        candidate = candidate.split()

        bleu1=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu2=sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu3=sentence_bleu(reference, candidate, weights=(0.333, 0.333, 0.333, 0))
        bleu4=sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        total_bleu1[ref_length] = total_bleu1[ref_length] + bleu1
        total_bleu2[ref_length] = total_bleu2[ref_length] + bleu2
        total_bleu3[ref_length] = total_bleu3[ref_length] + bleu3
        total_bleu4[ref_length] = total_bleu4[ref_length] + bleu4
        total_sample[ref_length] = total_sample[ref_length] + 1


        # print('ref:', reference)
        # print('hyp:', candidate)
        # print('bleu1:', bleu1)
        # print('bleu2:', bleu2)
        # print('bleu3:', bleu3)
        # print('bleu4:', bleu4)

total_bleu1 = total_bleu1 / total_sample * 100
total_bleu2 = total_bleu2 / total_sample * 100
total_bleu3 = total_bleu3 / total_sample * 100
total_bleu4 = total_bleu4 / total_sample * 100

# Smoothing
smoothing_factor=5
for i in range(max_length):
    total_bleu1[i] = total_bleu1[i:i+smoothing_factor].mean()
    total_bleu2[i] = total_bleu2[i:i+smoothing_factor].mean()
    total_bleu3[i] = total_bleu3[i:i+smoothing_factor].mean()
    total_bleu4[i] = total_bleu4[i:i+smoothing_factor].mean()

# Plotting the data
plt.rcParams['font.size'] = '20'

plt.figure(figsize=(10, 7))
plt.plot(total_bleu1.numpy()[:max_length], label='BLEU1')  # Convert torch tensor to NumPy array for plotting
plt.plot(total_bleu2.numpy()[:max_length], label='BLEU2')  # Convert torch tensor to NumPy array for plotting
plt.plot(total_bleu3.numpy()[:max_length], label='BLEU3')  # Convert torch tensor to NumPy array for plotting
plt.plot(total_bleu4.numpy()[:max_length], label='BLEU4')  # Convert torch tensor to NumPy array for plotting

plt.xlabel('Length', fontsize=20)
plt.ylabel('Scores', fontsize=20)
plt.grid(True)
plt.legend(fontsize="20", loc ="upper right")
plt.savefig(input_file+'.jpg')
plt.show()

plt.clf()
plt.figure(figsize=(10, 7))
plt.plot(total_sample.numpy()[:max_length], label='Total samples')  # Convert torch tensor to NumPy array for plotting
plt.xlabel('Length', fontsize=20)
plt.ylabel('Number of samples', fontsize=20)
plt.grid(True)
plt.savefig(input_file+'_num_samples.jpg')
plt.show()


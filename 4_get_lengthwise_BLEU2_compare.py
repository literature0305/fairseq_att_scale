import torch
import numpy
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

# Plotting the data
plt.rcParams['font.size'] = '20'
plt.figure(figsize=(10, 7))
max_length=120

input_files=['errlog012-2_decode_baseline', 'errlog012-2_decode_band-width-scaling']
for input_file in input_files:
    with open(input_file, 'r') as f:
        lines=f.readlines()

    total_bleu1=torch.zeros(1000)
    total_bleu2=torch.zeros(1000)
    total_bleu3=torch.zeros(1000)
    total_bleu4=torch.zeros(1000)
    total_sample=torch.ones(1000)


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

    label1=input_file+'BLEU1'
    label2=input_file+'BLEU2'
    label3=input_file+'BLEU3'
    label4=input_file+'BLEU4'
    # plt.plot(total_bleu1.numpy()[:max_length], label=label1)  # Convert torch tensor to NumPy array for plotting
    # plt.plot(total_bleu2.numpy()[:max_length], label=label2)  # Convert torch tensor to NumPy array for plotting
    # plt.plot(total_bleu3.numpy()[:max_length], label=label3)  # Convert torch tensor to NumPy array for plotting
    plt.plot(total_bleu4.numpy()[:max_length], label=label4)  # Convert torch tensor to NumPy array for plotting

    bin_width=10
    print('input_file:', input_file)
    bleu1_binwise=0
    bleu2_binwise=0
    bleu3_binwise=0
    bleu4_binwise=0
    for i in range(max_length):
        bin_num = (i+1)//bin_width
        if (i+1)%bin_width ==0:
            print('bin_num:', bin_num, 'bleu1:', bleu1_binwise / bin_width,  'bleu2:', bleu2_binwise / bin_width,  'bleu3:', bleu3_binwise / bin_width,  'bleu4:', bleu4_binwise / bin_width)
            bleu1_binwise=0
            bleu2_binwise=0
            bleu3_binwise=0
            bleu4_binwise=0
        bleu1_binwise = bleu1_binwise + total_bleu1[i]
        bleu2_binwise = bleu2_binwise + total_bleu2[i]
        bleu3_binwise = bleu3_binwise + total_bleu3[i]
        bleu4_binwise = bleu4_binwise + total_bleu4[i]
        

plt.xlabel('Length', fontsize=20)
plt.ylabel('Scores', fontsize=20)
plt.grid(True)
plt.legend(fontsize="20", loc ="upper right")
plt.savefig(input_file+'_compare.jpg')
plt.show()

# plt.clf()
# plt.figure(figsize=(10, 7))
# plt.plot(total_sample.numpy()[:max_length], label='Total samples')  # Convert torch tensor to NumPy array for plotting
# plt.xlabel('Length', fontsize=20)
# plt.ylabel('Number of samples', fontsize=20)
# plt.grid(True)
# plt.savefig(input_file+'_num_samples.jpg')
# plt.show()


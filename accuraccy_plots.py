import matplotlib.pyplot as plt
import re


def extract_accuracies(log_file_path):
    train_accs = []  
    val_accs = []    

   
    train_pattern = re.compile(r'\[Train\]\s+Epoch\s+\d+.*Train_Acc\s+([\d.]+)')
    val_pattern = re.compile(r'\[Val\]\s+Epoch\s+\d+.*Val_Acc\s+([\d.]+)')

 
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
            
                train_match = train_pattern.search(line)
                if train_match:
                    acc = train_match.group(1)
                    train_accs.append(float(acc))

         
                val_match = val_pattern.search(line)
                if val_match:
                    acc = val_match.group(1)
                    val_accs.append(float(acc))
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found at: {log_file_path}")

    
    if len(train_accs) != len(val_accs):
        raise ValueError(f"Mismatch in number of Train and Val entries: {len(train_accs)} Train vs {len(val_accs)} Val")

    return train_accs, val_accs


def plot_accuracies(train_accs, val_accs):
    plt.figure(figsize=(10, 6))

   
    x_axis = list(range(1, len(train_accs) + 1))

  
    plt.plot(x_axis, train_accs, label='Train Accuracy', marker='o', color='blue')
    plt.plot(x_axis, val_accs, label='Validation Accuracy', marker='s', color='red')

 
    plt.title('Train and Validation Accuracy Over Steps (Combined Runs)')
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

 
    plt.xticks(x_axis)


    output_path = '/kaggle/working/results/run_resnet50_imagenet_prune1/accuracy_plot.png'
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


log_file_path = '/kaggle/working/results/run_resnet50_imagenet_prune1/train_logger.log'


try:
    train_accs, val_accs = extract_accuracies(log_file_path)
    print("Extracted data:")
    print("Train Accuracies:", train_accs)
    print("Validation Accuracies:", val_accs)
    print("Steps:", list(range(1, len(train_accs) + 1)))
    plot_accuracies(train_accs, val_accs)
except Exception as e:
    print(f"Error: {e}")

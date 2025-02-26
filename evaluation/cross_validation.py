import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from .performance_metrics import evaluate_model
from models.wgan_gp import generate_samples
from evaluation.memory_record import print_memory_usage
import torch
from data_processing.data_loader import CustomDataset
from torch.utils.data import DataLoader
from models.wgan_gp import train_binary_classifier
from models.wgan_gp import WGAN_GP
from models.wgan_gp import MLPBinaryClassifier

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def divide_into_folds(data, labels, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    data = np.array(data)
    labels = np.array(labels)
    for train_indices, val_indices in kf.split(data):
        train_data, train_labels = data[train_indices], labels[train_indices]
        val_data, val_labels = data[val_indices], labels[val_indices]
        yield train_data, train_labels, val_data, val_labels

     

def perform_single_validation(generator, binary_classifier, num_samples, train_data, train_labels, val_data, val_labels, scaler, selected_samples=5000,train_batchsize=32, fold=1, model_params=None, train_params=None, sample_filter_threshold=0.1):
    accuracies, precisions, recalls, f1_scores = [], [], [], [] 
    if num_samples > 0:
        # using WGAN_GP to generate samples
        augmented_data = generate_samples(generator=generator, data=train_data, scaler=scaler,binary_classifier=binary_classifier, selected_samples=selected_samples, num_new_samples=num_samples,latent_dim=100,sample_filter_threshold=sample_filter_threshold)
        augmented_labels = [1 if augmented_data[:,0][i] >= 0.5 else 0 for i in range(len(augmented_data))]
        
        # concatenate the augmented data and the original training data
        scaler1 = StandardScaler()
        augmented_data = np.log1p(augmented_data)
        augmented_data = scaler1.fit_transform(augmented_data)
        
        train_data = np.concatenate((train_data, augmented_data))
        train_labels = np.concatenate((train_labels, augmented_labels))
    
    else :
        train_data = train_data
    # train and evaluate the model
    accuracy, precision, recall, f1 =  evaluate_model(train_data = train_data, train_labels=train_labels, val_data=val_data, val_labels=val_labels, num_samples = num_samples, fold=fold, model_params=model_params, train_params=train_params)

    # save the performance metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    return accuracies, precisions, recalls, f1_scores


def perform_cross_validation(data, labels, scaler,num_folds=5, num_samples_list=None, model_params=None, train_params=None, train_batchsize=32, sample_filter_threshold=0.1):  
    if num_samples_list is None:
        num_samples_list = [0, 100, 200, 300]  # number of samples to generate
    #new empty np array to store the results of each cross-validation
    results = {num_samples: [] for num_samples in num_samples_list}

    for fold, (train_data, train_labels, val_data, val_labels) in enumerate(divide_into_folds(data, labels, num_folds=num_folds), start=1):
        train_dataset=CustomDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
        
        fold_results = {num_samples: [] for num_samples in num_samples_list}
        #train generator
        WGAN = WGAN_GP(latent_dim=100,data_dim=data.shape[1],epochs=3800,lr=0.001)
        WGAN.train(train_loader)
        generator=WGAN.generator
        #train binary classifier
        binary_classifier = MLPBinaryClassifier(input_dim=data.shape[1]).to(device)
        train_binary_classifier(binary_classifier=binary_classifier,generator=generator,train_data=train_data,test_data=val_data)
        for num_samples in num_samples_list:
            accuracy, precision, recall, f1_score = perform_single_validation(generator=generator,
                                                                            binary_classifier=binary_classifier,
                                                                            num_samples=num_samples,
                                                                            train_data=train_data,
                                                                            train_labels=train_labels,
                                                                            val_data=val_data,
                                                                            val_labels=val_labels,
                                                                            scaler=scaler,
                                                                            selected_samples=num_samples,
                                                                            model_params=model_params,
                                                                            train_params=train_params,
                                                                            train_batchsize=train_batchsize,
                                                                            fold=fold,
                                                                            sample_filter_threshold=sample_filter_threshold
                                                                            )

            fold_results[num_samples].append((accuracy, precision, recall, f1_score))

        for num_samples in num_samples_list:
            results[num_samples].extend(fold_results[num_samples])

# Calculate mean and standard deviation
    final_results = []

    for num_samples in num_samples_list:
        accuracies = [result[0] for result in results[num_samples]]
        precisions = [result[1] for result in results[num_samples]]
        recalls = [result[2] for result in results[num_samples]]
        f1_scores = [result[3] for result in results[num_samples]]

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        mean_precision = np.mean(precisions)
        std_precision = np.std(precisions)

        mean_recall = np.mean(recalls)
        std_recall = np.std(recalls)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        final_results.append((num_samples, mean_accuracy, std_accuracy, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1))
    return final_results

        

def cross_validation_non_sample_generated(data, labels, num_folds=5,model_params=None, train_params=None,train_batchsize=32,val_batchsize=32):
    print('start cross validation...')
    print_memory_usage()
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    data = np.array(data)
    labels = np.array(labels)
    for train_indices, val_indices in kf.split(data):
        print("start training and evaluating model...")
        print_memory_usage()
        train_data, train_labels = data[train_indices], labels[train_indices]
        val_data, val_labels = data[val_indices], labels[val_indices]

        # train and evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(train_data, train_labels, val_data, val_labels,  model_params=model_params, train_params=train_params,train_batchsize=train_batchsize,val_batchsize=val_batchsize)

        # save the performance metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # calculate mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    # save the results in a tuple
    results = (mean_accuracy, std_accuracy, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1)

    print("training and evaluating model finished...")
    print_memory_usage()
    return results

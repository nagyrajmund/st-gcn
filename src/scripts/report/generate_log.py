from generate_figures import plot_multiple_diagrams, moving_average

#########
# Batch #
#########
file_names = ["logs/batchnorm_layer2_off_logs/train_acc.csv",
                "logs/batchnorm_layer2_off_logs/val_acc.csv"]
labels = ["Training accuracy", "Validation accuracy"]
plot_multiple_diagrams(file_names, labels, "Batch normalisation - only first layer", "img/batchnorm_layer2_off_acc.png")

file_names = ["logs/batchnorm_layer2_off_logs/train_loss.csv",
                "logs/batchnorm_layer2_off_logs/val_loss.csv"]
labels = ["Training loss", "Validation loss"]
plot_multiple_diagrams(file_names, labels, "Batch normalisation - only first layer", "img/batchnorm_layer2_off_loss.png")

############
# Residual #
############
file_names = ["logs/other/run-version_0-tag-val_acc.csv", "logs/other/run-version_2-tag-val_acc-2.csv"]
labels = ['Residual block', 'Without residual']
plot_multiple_diagrams(file_names, labels, 'Residual block on uni-labelling', 'img/unilabelling_res.png')

################
# Augmentation #
################
file_names = ['logs/other/run-version_4-tag-val_acc.csv', 'logs/other/run-version_22-tag-val_acc.csv']
labels = ['No data augmentation', 'Data augmentation']
plot_multiple_diagrams(file_names, labels, 'Validation accuracies with and without data augmentation', 'img/val_acc_data_aug.png')

#################
# Learning rate #
#################
file_names = ['logs/other/run-version_29-tag-val_acc.csv', 'logs/other/run-version_30-tag-val_acc.csv', 'logs/other/run-version_31-tag-val_acc.csv']
labels = ['0.01', '0.001', '0.0001']
plot_multiple_diagrams(file_names, labels, 'Validation accuracy for varying learning rate with dropout=0.25', 'img/dropout-0.25-val_loss.png')

file_names = ['logs/other/run-version_29-tag-train_loss.csv', 'logs/other/run-version_30-tag-train_loss.csv', 'logs/other/run-version_31-tag-train_loss.csv']
labels = ['0.01', '0.001', '0.0001']
plot_multiple_diagrams(file_names, labels, 'Training loss for varying learning rate with dropout=0.25', 'img/dropout-0.25-train_loss.png')

#############################################
# Dropout, learning rate, data augmentation #
#############################################
title = 'Val. acc. for dropout 0.5 with varying learning rates (with data augmentation)'
file_names = ['logs/other/run-version_22-tag-val_acc.csv', 'logs/other/run-version_23-tag-val_acc.csv']
plot_multiple_diagrams(file_names, ['lr=0.01', 'lr=0.0001'], title, 'img/dropout-0.5_lr_acc.png')

title = 'Val. loss for dropout 0.5 with varying learning rates (with data augmentation)'
file_names = ['logs/other/run-version_22-tag-val_loss.csv', 'logs/other/run-version_23-tag-val_loss.csv']
plot_multiple_diagrams(file_names, ['lr=0.01', 'lr=0.0001'], title, 'img/dropout-0.5_lr_loss.png')
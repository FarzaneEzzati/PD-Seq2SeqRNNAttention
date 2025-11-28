import pickle
import time
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from Seq2Seq import Seq2Seq
from DataPreprocessor import get_train_valid, get_batches
from Loss import QuantileLoss

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Early stopping
class EarlyStopping:
    def __init__(self, patience, min_delta, name_to_save):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.save_path = f'Models/{name_to_save}.pt'

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1

        return self.counter >= self.patience


# Seq2Seq class
class Seq2SeqModel:
    def __init__(self, args, name):
        torch.backends.cudnn.enabled = False
        # Build Seq2Seq model
        self.model = Seq2Seq(
            input_seq_len=args['input_seq_len'],
            output_seq_len=args['output_seq_len'],
            input_dim=args['input_dim'],
            output_dim=args['output_dim'],
            n_layers=args['n_layers'],
            enc_hidden_dim=args['enc_hidden_dim'],
            dec_hidden_dim=args['dec_hidden_dim'],
            dropout=args['dropout'])
        self.model = self.model.to(device)
        self.name_to_save = name

        print(f'The model has {count_parameters(self.model):,} trainable parameters')
        print(f'Initial learning rate is {args["initial_lr"]}')

        # Define loss (criterion) and optimizer
        self.criterion = QuantileLoss(args['quantiles'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=args["initial_lr"], amsgrad=True, weight_decay=1e-3)

        # Early stopping class
        self.early_stopping = EarlyStopping(
            patience=args['patience'],
            min_delta=args['min_delta'],
            name_to_save=self.name_to_save)

        # Private args
        self.args = args


    def start_training(self, X, y):
        torch.backends.cudnn.enabled = False  # avoiding reproducibility, debugging, and compatibility issues

        # Train mode
        self.model.train()

        # Get train and valid data
        train_X, train_y, valid_X, valid_y = get_train_valid(X, y, valid_years=self.args['valid_years'])

        # Get quantiles data
        quantiles = self.criterion.quantiles

        # history of loss, MAE, and certainty
        train_loss_hist, val_loss_hist = [], []
        train_qloss_hist, val_qloss_hist = [], []
        train_mae_hist, val_mae_hist = [], []
        train_mape_hist, val_mape_hist = [], []
        train_cer_hist, val_cer_hist = [], []

        for epoch in range(self.args['n_epochs']):
            # temporary list of loss, MAE, and certainty
            train_loss_list, train_qloss_list = [], []
            train_mae_list, train_mape_list, train_cer_list = [], [], []

            # Decay the learning rate each epoch. optimizer is updated.
            optimizer, lr = sqrt_lr_scheduler(self.optimizer, epoch, self.args['initial_lr'])

            # Decay the tf-ratio
            tf_ratio = sqrt_tf_scheduler(epoch, self.args['initial_tf_ratio'])

            # Training batches
            batches_x, batches_y = get_batches(X=train_X, y=train_y,
                                               batch_size=self.args['batch_size_train'],
                                               n_batches=self.args['n_batches_train'],
                                               input_seq_len=self.args['input_seq_len'],
                                               output_seq_len=self.args['output_seq_len'],
                                               input_dim=self.args['input_dim'])
            start = time.time()
            for batch, (x, y) in enumerate(zip(batches_x, batches_y)):
                # Feed tensor data
                x, y = map(lambda var: torch.from_numpy(var).to(device), (x, y))  # x: [batch_size, seq_len, input_dim]
                optimizer.zero_grad()
                preds = self.model(x, y, tf_ratio=tf_ratio)

                # Calculate loss
                y_reshaped = y.unsqueeze(2).repeat(1, 1, self.args['output_dim'])  # [batch_size, seq_len, output_dim]
                loss, quantile_loss = self.criterion(preds, y_reshaped)

                # Store loss
                train_loss_list.append(loss.item())
                train_qloss_list.append(quantile_loss)

                # Store MAE and Certainty
                train_mae_list.append(
                    calculate_mae(preds, y_reshaped, self.args['target_mean'], self.args['target_std']))
                train_mape_list.append(
                    calculate_mape(preds, y_reshaped, self.args['target_mean'], self.args['target_std']))
                train_cer_list.append(
                    calculate_certainty(preds, y_reshaped))

                # Backpropagation
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['clip'])  # clip grad if exceeds a threshold
                optimizer.step()
            finish = time.time()
            # Eval mode
            self.model.eval()
            val_loss_list, val_qloss_list, val_mae_list, val_mape_list, val_cer_list = self.evaluate(valid_X=valid_X, valid_y=valid_y)

            # Train mode
            self.model.train()  # turn on model training

            # Save epoch losses
            train_loss_hist.append(np.mean(train_loss_list))
            val_loss_hist.append(np.mean(val_loss_list))
            train_qloss_hist.append(np.mean(train_qloss_list, axis=0))
            val_qloss_hist.append(np.mean(val_qloss_list, axis=0))

            # Save epoch MAE and Certainty
            train_mae_hist.append(np.mean(train_mae_list))
            train_mape_hist.append(np.mean(train_mape_list))
            train_cer_hist.append(np.mean(train_cer_list))
            val_mae_hist.append(np.mean(val_mae_list))
            val_mape_hist.append(np.mean(val_mape_list))
            val_cer_hist.append(np.mean(val_cer_list))

            # Print epoch results
            print_results(epoch, self.args['n_epochs'], finish-start,
                          train_loss_list, val_loss_list, train_cer_list, val_cer_list)

            # Check early stop
            if self.early_stopping(np.mean(val_loss_list), self.model) or epoch + 1 == self.args['n_epochs']:
                # Plot losses
                plot_losses(
                    train_loss_hist, val_loss_hist, name=self.name_to_save + "-Loss")
                plot_qlosses(
                    train_qloss_hist, val_qloss_hist, name=self.name_to_save + "-Qloss", quantiles=quantiles)
                plot_accuracy(
                    train_mae_hist, val_mae_hist, type='MAE', name=self.name_to_save + "-MAE")
                plot_accuracy(
                    train_mape_hist, val_mape_hist, type='MAPE', name=self.name_to_save + "-MAPE")
                plot_certainty(
                    train_cer_hist, val_cer_hist, name=self.name_to_save + "-Certainty")

                # Save model at the end
                if epoch + 1 == self.args['n_epochs']:
                    torch.save(self.model.state_dict(), f'Models/{self.name_to_save}.pt')

                # Save loss and target stats
                with open(f'Data/({self.args["location"]})Performance.pkl', 'wb') as handle:
                    pickle.dump(
                        [train_loss_hist, val_loss_hist, train_mae_hist, val_mae_hist, train_cer_hist, val_cer_hist],
                        handle)

                # Break epoch loop
                break

    def evaluate(self, valid_X, valid_y):
        val_losses = []
        val_qlosses = []
        val_mae = []
        val_mape = []
        val_cer = []
        batches_x, batches_y = get_batches(X=valid_X, y=valid_y,
                                           batch_size=self.args['batch_size_val'],
                                           n_batches=self.args['n_batches_val'],
                                           input_seq_len=self.args['input_seq_len'],
                                           output_seq_len=self.args['output_seq_len'],
                                           input_dim=self.args['input_dim'])
        for batch, (val_x, val_y) in enumerate(zip(batches_x, batches_y)):
            with torch.no_grad():
                # Predict
                val_x, val_y = map(lambda var: torch.from_numpy(var).to(device), (val_x, val_y))
                val_pred = self.model(val_x, val_y, tf_ratio=0)
                val_y_reshaped = val_y.unsqueeze(2).repeat(1, 1, self.args['output_dim'])  # [batch_size, seq_len, output_dim]
                val_loss, val_quantile_loss = self.criterion(val_pred, val_y_reshaped)
                val_losses.append(val_loss.item())
                val_qlosses.append(val_quantile_loss)

                # MAE and Certainty
                val_mae.append(
                    calculate_mae(val_pred, val_y_reshaped,  self.args['target_mean'],  self.args['target_std']))
                val_mape.append(
                    calculate_mape(val_pred, val_y_reshaped, self.args['target_mean'], self.args['target_std']))
                val_cer.append(
                    calculate_certainty(val_pred, val_y_reshaped))

        return val_losses, val_qlosses, val_mae, val_mape, val_cer


    def predict(self, X_inf, nn_param=None):
        if nn_param is not None:
            self.model.load_state_dict(torch.load(nn_param))
        self.model.eval()

        # Eval mode
        self.model.eval()
        X = torch.tensor(X_inf.astype('float32')).unsqueeze(0)
        y = torch.zeros(24).unsqueeze(0)

        # Disable gradient computation for inference
        with torch.no_grad():
            inference = self.model(X, y, 0)
        return inference


def calculate_mae(preds, targets, target_mean, target_std):
    with torch.no_grad():
        trans_preds = preds * target_std + target_mean
        trans_targets = targets * target_std + target_mean
        mae = torch.mean(torch.abs(trans_preds[:, 1] - trans_targets[:, 1]))
    return mae


def calculate_mape(preds, targets, target_mean, target_std):
    with torch.no_grad():
        epsilon = torch.full_like(targets, fill_value=0.00001)
        trans_preds = preds * target_std + target_mean
        trans_targets = targets * target_std + target_mean
        mape = torch.abs((trans_preds - trans_targets) / (trans_targets + epsilon))
        mape = 100 * torch.mean(mape[:, 1])
    return mape


def calculate_certainty(preds, targets):
    certainty = torch.mean(((targets[:, :, 1] <= preds[:, :, 2]) & (targets[:, :, 1] >= preds[:, :, 0])).float())
    return certainty.item()


def sqrt_lr_scheduler(optimizer, epoch, initial_lr):
    # Decay learning rate by square root of the epoch
    if epoch in [0, 1]:  # first 2 epochs
        lr = initial_lr
    else:
        lr = initial_lr / np.sqrt(epoch + 2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # update the learning for all groups of params. could be selective also.
    return optimizer, lr


def sqrt_tf_scheduler(epoch, initial_tf_ratio):
    if epoch in [0, 1]:  # first 2 epochs
        tf_ratio = initial_tf_ratio
    else:
        tf_ratio = initial_tf_ratio / np.sqrt(epoch + 2)
    return tf_ratio


def plot_losses(train_loss_hist, val_loss_hist, name):
    plt.plot(train_loss_hist, label='train loss')
    plt.plot(val_loss_hist, label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average Quantile Loss')
    plt.legend()
    plt.savefig(f'Plots/{name}.jpg', dpi=300)
    plt.close()


def plot_qlosses(train_qloss_hist, val_qloss_hist, name, quantiles):
    _, axs = plt.subplots(1, 2, figsize=(10, 4))

    for i, q in enumerate(quantiles):
        axs[0].set_title('Train')
        axs[0].plot(np.array(train_qloss_hist)[:, i], label=f'Quantile {q:0.2f}')
        axs[1].set_title('Validation')
        axs[1].plot(np.array(val_qloss_hist)[:, i], label=f'Quantile {q:0.2f}')
    for ax in axs:
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Quantile Loss')
        ax.legend()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(f'Plots/{name}.jpg', dpi=300)
    plt.close()


def plot_accuracy(train_mae_hist, val_mae_hist, type, name):
    plt.plot(train_mae_hist, label='train')
    plt.plot(val_mae_hist, label='valid')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.savefig(f'Plots/{name}.jpg', dpi=300)
    plt.close()


def plot_certainty(train_cer_hist, val_cer_hist, name):
    plt.plot(train_cer_hist, label='train')
    plt.plot(val_cer_hist, label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Certainty')
    plt.legend()
    plt.savefig(f'Plots/{name}.jpg', dpi=300)
    plt.close()


def print_results(epoch, n_epochs, duration, train_loss_list, val_loss_list, train_cer_list, val_cer_list):
    print(f'Epoch: {epoch + 1}/{n_epochs}  '
          f'{duration:0.2f}s  '
          f'T-Loss: {np.mean(train_loss_list):.4f}  '
          f'V-Loss: {np.mean(val_loss_list):.4f}   '
          f'T-Cert.: {np.mean(train_cer_list):.2f}   '
          f'V-Cert.: {np.mean(val_cer_list):.2f}')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

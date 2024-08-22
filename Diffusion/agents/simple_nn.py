import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import os
from agents.callback import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.callbacks import Callback, EarlyStopping


def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    return (data - mean) / std


def warmup_lr_scheduler(optimizer, warmup_epochs, start_lr, target_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (target_lr - start_lr) / warmup_epochs * epoch + start_lr
        else:
            return target_lr / optimizer.defaults['lr']

    return LambdaLR(optimizer, lr_lambda)


def discretize_action_to_control_mode_E2E(action):
    # Your action discretization logic here
    action_norm = (action + 1) / 2
    if 1 / 6 > action_norm >= 0:
        control_mode = 0
        friction_state = 1  # left finger high friction
    elif 2 / 6 > action_norm >= 1 / 6:
        control_mode = 1
        friction_state = 1
    elif 3 / 6 > action_norm >= 2 / 6:
        control_mode = 2
        friction_state = -1
    elif 4 / 6 > action_norm >= 3 / 6:
        control_mode = 3
        friction_state = -1
    elif 5 / 6 > action_norm >= 4 / 6:
        control_mode = 4
        friction_state = 0
    else:
        assert 1 >= action_norm >= 5 / 6, f"Wrong action size: {action, action_norm}"
        control_mode = 5
        friction_state = 0
    # print(action_norm)
    return friction_state, control_mode


def terminate_classify_data(demo_path, demo_path_norm):
    with open(demo_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(demo_path_norm, 'rb') as f:
        dataset_4_norm = pickle.load(f)

    # print(dataset['observations'].shape)
    # assert dataset['observations'].shape[1] == 18, f"check: {dataset['observations'].shape}"

    if dataset['desired_goals'].shape[1] == 11:
        dataset['desired_goals'] = np.array([sub_array[:-4] for sub_array in dataset['desired_goals']])
    if dataset_4_norm['desired_goals'].shape[1] == 11:
        dataset_4_norm['desired_goals'] = np.array([sub_array[:-4] for sub_array in dataset_4_norm['desired_goals']])
    # if dataset['desired_goals'].shape[1] == 9:
    #     dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
    # if dataset_4_norm['desired_goals'].shape[1] == 9:
    #     dataset_4_norm['desired_goals'] = np.array(
    #         [sub_array[:-2] for sub_array in dataset_4_norm['desired_goals']])

    obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
    obs_std = np.std(dataset_4_norm['observations'], axis=0)
    obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
    dataset['observations'] = obs_norm

    goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)
    goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)
    goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
    dataset['desired_goals'] = goal_norm

    assert goal_norm.shape[1] == 9, f"check: {goal_norm.shape[1]}"  # 7

    robot_qpos = dataset['observations'][:, 0:4]  # 4
    object_qpos = dataset['observations'][:, 8:15]  # 7
    inputs = np.hstack((robot_qpos, object_qpos, goal_norm))  # 18

    # assert inputs.shape[1] == 18, f"check: {inputs.shape[1]}"

    # print(inputs[0])

    targets = np.expand_dims(dataset['terminals'].copy(), axis=1)

    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets.squeeze()).long()

    # Split data into training and validation sets
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2,
                                                                            random_state=42)

    train_dataset = {'input': train_inputs, 'target': train_targets}
    val_dataset = {'input': val_inputs, 'target': val_targets}

    return train_dataset, val_dataset


def control_mode_classify_data(demo_path, demo_path_norm):
    action_mode = []
    with open(demo_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(demo_path_norm, 'rb') as f:
        dataset_4_norm = pickle.load(f)

    if dataset['desired_goals'].shape[1] == 11:
        dataset['desired_goals'] = np.array([sub_array[:-4] for sub_array in dataset['desired_goals']])
    if dataset_4_norm['desired_goals'].shape[1] == 11:
        dataset_4_norm['desired_goals'] = np.array([sub_array[:-4] for sub_array in dataset_4_norm['desired_goals']])
    # if dataset['desired_goals'].shape[1] == 9:
    #     dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
    # if dataset_4_norm['desired_goals'].shape[1] == 9:
    #     dataset_4_norm['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset_4_norm['desired_goals']])

    obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
    obs_std = np.std(dataset_4_norm['observations'], axis=0)
    obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
    dataset['observations'] = obs_norm

    goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)
    goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)
    goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
    dataset['desired_goals'] = goal_norm

    assert goal_norm.shape[1] == 9, f"check: {goal_norm.shape[1]}"

    robot_qpos = dataset['observations'][:, 0:4]
    object_qpos = dataset['observations'][:, 8:15]
    inputs = np.hstack((robot_qpos, object_qpos, goal_norm))

    print(inputs.shape, robot_qpos.shape, object_qpos.shape, goal_norm.shape)

    for i, item in enumerate(dataset['actions']):
        action_mode.append(discretize_action_to_control_mode_E2E(item[1])[1])
    action_mode = np.array(action_mode)
    print(np.shape(action_mode))

    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(action_mode.squeeze()).long()

    # Split data into training and validation sets
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2,
                                                                            random_state=42)

    train_dataset = {'input': train_inputs, 'target': train_targets}
    val_dataset = {'input': val_inputs, 'target': val_targets}

    return train_dataset, val_dataset


def plot_target(data):
    targets = data['target'].numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(targets, 'o', markersize=2)
    plt.xlabel('Index')
    plt.ylabel('Target')
    plt.title('Target vs. Index')
    plt.show()


def plot_target_prediction(targets, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(targets)), targets, color='orange', s=10, label='Actual Targets')
    plt.scatter(range(len(predictions)), predictions, color='blue', s=10, label='Predictions')
    plt.xlabel('Index')
    plt.ylabel('Target')
    plt.title('Target vs. Predictions')
    plt.legend()


class TransformerTabNet:
    def __init__(self, input_size, num_classes):
        self.model = TabNetClassifier(
            input_dim=input_size,
            output_dim=num_classes,
            n_d=32,  # Increasing hidden dimension size
            n_a=32,  # Increasing attention dimension size
            n_steps=5,  # Increasing the number of decision steps
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            lambda_sparse=1e-4,  # Increased regularization
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),  # Adjusting learning rate
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax'
        )

    def fit(self, X_train, y_train, X_val, y_val, max_epochs=200, patience=20, batch_size=256):
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy', 'balanced_accuracy', 'logloss'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=batch_size // 2,
            num_workers=0,
            drop_last=False
        )

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes, dropout_prob=0.5):
        """
        Parameters:
        - input_size: int, the number of input features
        - num_classes: int, the number of output classes
        - hidden_sizes: list of int, sizes of the hidden layers
        """
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))  # Add dropout layer

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, predict, target):
        return self.criterion(predict, target.long())


class EnhancedNN(nn.Module):
    def __init__(self, input_size, num_classes, layers=[512, 256, 128]):
        super(EnhancedNN, self).__init__()
        self.layers = nn.ModuleList()
        in_features = input_size

        for layer_size in layers:
            self.layers.append(nn.Linear(in_features, layer_size))
            self.layers.append(nn.BatchNorm1d(layer_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.5))
            in_features = layer_size

        # Output layer
        self.layers.append(nn.Linear(in_features, num_classes))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, predict, target):
        return self.criterion(predict, target.long())


def train_and_evaluate_v4(model, model_name, train_data, test_data, num_epochs=1000, warmup_epochs=10, patience=100,
                          use_warmup=True, base_save_path="models/", batch_size=256, use_scheduler=True):
    save_path = os.path.join(base_save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    losses = []
    val_accuracies = []

    if isinstance(model, TransformerTabNet):
        checkpoint = ModelCheckpoint(save_path=save_path, save_interval=10, monitor='valid_logloss', mode='min')
        model.model.fit(
            X_train=train_data['input'].numpy(), y_train=train_data['target'].numpy(),
            eval_set=[(train_data['input'].numpy(), train_data['target'].numpy()),
                      (test_data['input'].numpy(), test_data['target'].numpy())],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy', 'balanced_accuracy', 'logloss'],
            max_epochs=100,
            patience=patience,
            batch_size=batch_size,
            callbacks=[checkpoint],
        )
        best_model_path = os.path.join(save_path, 'best_model.zip')
        model.model.load_model(best_model_path)
        history = checkpoint.get_history()
        losses = history['valid_logloss']
        val_accuracies = history['valid_accuracy']
        return model, losses, val_accuracies
    else:
        # Convert to torch tensors
        inputs = train_data['input'].clone().detach().to(torch.float32).to(device)
        targets = train_data['target'].clone().detach().to(torch.long).to(device)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_acc = 0.0
        patience_counter = 0

        # Warm-up scheduler
        if use_scheduler and use_warmup:
            warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, start_lr=0.0001, target_lr=learning_rate)

        # CosineAnnealingWarmRestarts scheduler after warm-up
        if use_scheduler:
            main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                outputs = model(batch_inputs).squeeze(1)
                loss = model.loss(outputs, batch_targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_inputs.size(0)

            epoch_loss /= len(dataset)

            # Step the warm-up scheduler if in warm-up phase and warm-up is enabled
            if use_scheduler:
                if use_warmup and epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

            model.eval()
            with torch.no_grad():
                outputs_test = model(test_data['input']).squeeze(1)
                test_loss = model.loss(outputs_test, test_data['target'].long())
                losses.append(test_loss.item())
                predictions = torch.argmax(outputs_test, axis=1)
                acc = f1_score(test_data['target'], predictions, average='macro')
                val_accuracies.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_model_wts = model
                torch.save(model, os.path.join(save_path, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                torch.save(model, os.path.join(save_path, f'model_epoch_{epoch}.pth'))

            if epoch % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Training Loss: {epoch_loss:.4f}, '
                      f'Validation Loss: {test_loss.item():.4f}, '
                      f'Validation F1 Score: {acc:.4f}, '
                      f'Patience counter: {patience_counter}')

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        torch.load(os.path.join(save_path, 'best_model.pth'))
        return model, losses, val_accuracies


def train_and_evaluate_v4_robust(model, model_name, train_data, test_data, num_epochs=1000, warmup_epochs=10,
                                 patience=100,
                                 use_warmup=True, base_save_path="models/", batch_size=256, use_scheduler=True,
                                         noise_level=0.1, dropout_prob=0.5):
    save_path = os.path.join(base_save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    losses = []
    val_accuracies = []

    if isinstance(model, TransformerTabNet):
        checkpoint = ModelCheckpoint(save_path=save_path, save_interval=10, monitor='valid_logloss', mode='min')
        model.model.fit(
            X_train=train_data['input'].numpy(), y_train=train_data['target'].numpy(),
            eval_set=[(train_data['input'].numpy(), train_data['target'].numpy()),
                      (test_data['input'].numpy(), test_data['target'].numpy())],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy', 'balanced_accuracy', 'logloss'],
            max_epochs=num_epochs,
            patience=patience,
            batch_size=batch_size,
            callbacks=[checkpoint],
        )
        best_model_path = os.path.join(save_path, 'best_model.zip')
        model.model.load_model(best_model_path)
        history = checkpoint.get_history()
        losses = history['valid_logloss']
        val_accuracies = history['valid_accuracy']
        return model, losses, val_accuracies
    else:
        # Convert to torch tensors
        inputs = train_data['input'].clone().detach().to(torch.float32).to(device)
        targets = train_data['target'].clone().detach().to(torch.long).to(device)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_acc = 0.0
        patience_counter = 0

        # Warm-up scheduler
        if use_scheduler and use_warmup:
            warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, start_lr=0.0001, target_lr=learning_rate)

        # CosineAnnealingWarmRestarts scheduler after warm-up
        if use_scheduler:
            main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                # Add Gaussian noise to inputs
                noise = torch.normal(0, noise_level, size=batch_inputs.size()).to(batch_inputs.device)
                batch_inputs_noisy = batch_inputs + noise

                # Forward pass with noise
                outputs = model(batch_inputs_noisy).squeeze(1)

                # Apply Dropout during training
                outputs = F.dropout(outputs, p=dropout_prob, training=True)

                loss = model.loss(outputs, batch_targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_inputs.size(0)

            epoch_loss /= len(dataset)

            # Step the warm-up scheduler if in warm-up phase and warm-up is enabled
            if use_scheduler:
                if use_warmup and epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

            model.eval()
            with torch.no_grad():
                outputs_test = model(test_data['input']).squeeze(1)
                test_loss = model.loss(outputs_test, test_data['target'].long())
                losses.append(test_loss.item())
                predictions = torch.argmax(outputs_test, axis=1)
                acc = f1_score(test_data['target'], predictions, average='macro')
                val_accuracies.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_model_wts = model
                torch.save(model, os.path.join(save_path, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                torch.save(model, os.path.join(save_path, f'model_epoch_{epoch}.pth'))

            if epoch % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Training Loss: {epoch_loss:.4f}, '
                      f'Validation Loss: {test_loss.item():.4f}, '
                      f'Validation F1 Score: {acc:.4f}, '
                      f'Patience counter: {patience_counter}')

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        torch.load(os.path.join(save_path, 'best_model.pth'))
        return model, losses, val_accuracies


def test_prediction(model, model_name, test_data, save_path):
    if model_name == 'TransformerTabNet':
        best_model_path = os.path.join(save_path, model_name, 'best_model.zip')
        model.model.load_model(best_model_path)
        predictions = model.predict(test_data['input'].numpy())
        predictions = torch.tensor(predictions)
    else:
        best_model_path = os.path.join(save_path, model_name, 'best_model.pth')
        model = torch.load(best_model_path)
        model.eval()
        with (torch.no_grad()):
            #            input = np.array([-0.67667514, -0.98206216,  0.7821281,  -0.791338 ,   0.6824057,   1.3751259,
            # -1.9584274  , 0.8017153  , 3.245195  ,  1.4196168  , 0.728711 ,  -1.5444695,
            # -0.60912234 , 0.99969405 ,-1.0615209 , -1.4818825  , 1.238953 ,   1.4724548,
            #  1.9728364 , -0.8727226 ])
            #            input = torch.from_numpy(input).float().unsqueeze(0)
            #            print(input.shape)
            input = test_data['input']
            outputs_test = model(input)
            predictions = torch.argmax(outputs_test, axis=1)

    # print(test_data['input'].shape)
    acc = f1_score(test_data['target'], predictions, average='macro')
    print(f"Accuracy for {model_name}: {acc:.4f}")
    print("Prediction Visualization")
    plot_target_prediction(test_data['target'], predictions)

    image_path = os.path.join(save_path, model_name, 'prediction.png')
    plt.savefig(image_path)
    plt.show()
    plt.close()


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':

    # Parameters

    # base_save_path_ = 'models_terminate_cube_for_real'
    # base_save_path_ = 'models_operating_mode_cube_for_real'

    # base_save_path_ = "models_terminate_cube_cylinder"
    base_save_path_ = "models_operating_mode_cube_cylinder_robust"

    # base_save_path_ = "models_terminate_three_cylinder"

    # base_save_path_ = "models_terminate_cube_1cm"
    # base_save_path_ = "models_6_operating_mode_cube_1cm_512_4"

    # base_save_path_ = "models_terminate_cube_2cm"
    # base_save_path_ = "models_operating_mode_cube_2cm"

    # base_save_path_ = "models_terminate_mixed_object"
    # base_save_path_ = "models_operating_mode_mixed_object"

    # base_save_path_ = "test"
    test = False
    num_classes_ = 4  # Number of output classes. terminate: 2. operate mode: 4.
    num_epochs = 500  # Number of epochs
    learning_rate = 0.001  # Learning rate
    ' original '
    mid_layer = 256  # Number of neurons per layer
    out_layer = 64
    num_layers = 3  # Number of layers
    batch_size_ = 256

    ' Tune '
    # mid_layer = 1024  # Number of neurons per layer
    # out_layer = 512
    # num_layers = 4  # Number of layers
    # batch_size_ = 1024

    scheduler = True
    warmup = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: ", device)

    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_slide'

    ' Cube '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube/train_10k_cube_for_real'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube/train_10k_cube_slide_for_real'

    ' Three Cylinder '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_three_cylinder_slide'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/three_cylinder/train_10k_three_cylinder_noObjVel.pkl'

    ' Cube Cylinder '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_10k_cube_cylinder_noObjVel.pkl'

    demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_10k_cube_cylinder_noObjVel_forReal'

    ' Cube 1cm '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_slide'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel.pkl'

    ' Cube 2cm '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel_slide'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel.pkl'

    ' Mixed Object '
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel_slide'
    # demo_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel.pkl'

    'Terminate'
    # train_data = terminate_classify_data(demo_path, demo_path)
    # test_data = terminate_classify_data(demo_path_test, demo_path)
    # train_data, test_data = terminate_classify_data(demo_path, demo_path)

    'Control Mode'
    # test_data = control_mode_classify_data(demo_path_test)
    # train_data = control_mode_classify_data(demo_path)
    train_data, test_data = control_mode_classify_data(demo_path, demo_path)

    input_size_ = train_data['input'].shape[1]

    # Define models with consistent layer size and number of layers
    models = {
        'SimpleNN': SimpleNN(input_size_, num_classes_, hidden_sizes=[mid_layer] * (num_layers - 1) + [out_layer]),
        # 'EnhancedNN': EnhancedNN(input_size_, num_classes_, layers=[mid_layer] * (num_layers - 1) + [out_layer]),
        # 'TransformerTabNet': TransformerTabNet(input_size_, num_classes_)
    }

    results = {}

    for name, model in models.items():
        print("Warm up: ", warmup)
        print('-' * 50)
        print(f'\n{name} Architecture:')
        print(model)
        print('-' * 50)

        print(f'Training {name}...')
        if name != 'TransformerTabNet':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = None  # No external optimizer needed for TransformerTabNet
        if test:
            test_prediction(model, name, test_data, base_save_path_)
        else:
            # trained_model, losses, val_accuracies = train_and_evaluate_v4(model,
            #                                                               name,
            #                                                               train_data,
            #                                                               test_data,
            #                                                               num_epochs,
            #                                                               batch_size=batch_size_,
            #                                                               patience=100,
            #                                                               use_warmup=warmup,
            #                                                               use_scheduler=scheduler,
            #                                                               base_save_path=base_save_path_,
            #                                                               )
            trained_model, losses, val_accuracies = train_and_evaluate_v4_robust(model,
                                                                                 name,
                                                                                 train_data,
                                                                                 test_data,
                                                                                 num_epochs,
                                                                                 batch_size=batch_size_,
                                                                                 patience=100,
                                                                                 use_warmup=warmup,
                                                                                 use_scheduler=scheduler,
                                                                                 base_save_path=base_save_path_,
                                                                                 )
            results[name] = {'model': trained_model, 'losses': losses, 'val_accuracies': val_accuracies}

            # Plot the loss curves for each model
            for name, result in results.items():
                losses = result['losses']
                loss_var = np.var(losses)

                plt.plot(range(len(losses)), losses, label=f'{name} Validation Loss')
                plt.fill_between(range(len(losses)),
                                 np.array(losses) - np.sqrt(loss_var),
                                 np.array(losses) + np.sqrt(loss_var),
                                 alpha=0.2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss Curve with Variance for Each Model')
            plt.legend()
            image_path = os.path.join(base_save_path_, 'LossCurve.png')
            plt.savefig(image_path)
            plt.show()

            # Plot the validation F1 score curves for each model
            for name, result in results.items():
                plt.plot(range(len(result['val_accuracies'])), result['val_accuracies'],
                         label=f'{name} Validation F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Validation F1 Score for Each Model')
            plt.legend()
            image_path = os.path.join(base_save_path_, 'ValidationF1Score.png')
            plt.savefig(image_path)
            plt.show()

            best_model_name = max(results, key=lambda name: max(results[name]['val_accuracies']))
            best_f1_score = max(results[best_model_name]['val_accuracies'])
            print(f"Best Model: {best_model_name} with F1 Score: {best_f1_score}")

            test_prediction(model, name, test_data, base_save_path_)

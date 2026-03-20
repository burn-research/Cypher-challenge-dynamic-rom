'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import numpy as np
import aPrioriDNS as ap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_manager import DataManager
from utils import set_global_seed, product, mute_print

# ========== Logarithmic scaler used for some variables (e.g. gradients)
# Log + MinMax scaler pipeline for column 1
log_minmax_pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p, validate=True)),  # log(1 + x)
    ('minmax', MinMaxScaler())                              # scale to [0, 1]
])

# =============================== MODEL =======================================
class model(nn.Module):
    def __init__(self, verbose=True):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        super().__init__() # default initialization of the inherited nn.Module class
        
        # Set general purpose attributes
        self.is_trained=False
        self.seed=42
        self.alpha_t_scaling = 1e4
        
        # Set nn architecture parameters
        self.input_size = 4
        self.hidden_size = 128
        self.num_layers = 5
        self.output_size = 1
        
        self.n_epochs = 100
        self.test_size = 0.2
        
        # Initialize seed for reproducibility
        set_global_seed(self.seed)
        
        # Create the neural network's layers
        if verbose:
            print("Creating Neural Network's layers")
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size)) # Add the first input layer. 
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size)) # Add hidden layers
        self.layers.append(nn.Linear(self.hidden_size, self.output_size)) # add output layer
        
        # # Initialize weights
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_uniform_(layer.weight)  # or kaiming_uniform_, etc.
        #         nn.init.zeros_(layer.bias) # normal_(layer.bias, 0, 1)


    def preprocess(self, data_folder, verbose=True):
        M = DataManager(data_folder)
        if verbose:
            print ('\nKeys in data dictionary:')
            print (M.data_dict.keys())
        
        M.build_training_data(['T', 'C', 'RHO', 'C_grad'], ['TAU_C_X', 'TAU_C_Y', 'TAU_C_Z', 'RHO', 'C_grad_X', 'C_grad_Y', 'C_grad_Z'])
        
        if verbose:
            print(f'Successfully preprocessed the data in folder {data_folder}')
            print(f'Size of the X vector: {M.X.shape}')
            print(f'Size of the Y vector: {M.Y.shape}')
            print('Scaling the data...')
            
        self.scaler = ColumnTransformer(
                            transformers=[
                                ('minmax_0', MinMaxScaler(), [0]),          # MinMax for column 0
                                ('minmax_1', MinMaxScaler(), [1]),
                                ('minmax_2', MinMaxScaler(), [2]),
                                ('log_minmax_3', log_minmax_pipeline, [3])  # Log + MinMax for column 1
                            ]
                        )
        
        X_scaled = self.scaler.fit_transform(X=M.X)
        
        if verbose:
            print('Splitting data between training and validation...')
        X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, M.Y, test_size=self.test_size)
        
        
        D = dict()
        D["X"] = torch.tensor(X_train, dtype=torch.float32)
        D["Y"] = torch.tensor(Y_train, dtype=torch.float32) # Y is not scaled yet here
        D["X_val"] = torch.tensor(X_val, dtype=torch.float32)
        D["Y_val"] = torch.tensor(Y_val, dtype=torch.float32) # Y is not scaled yet here
        
        
        if verbose:
            print('This version of the code returns a dictionary with the actual X and Y pairs')
        
        return D

    def fit(self, D, verbose=True):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        if verbose:
            print('Trying to perform a forward pass for debugging purposes...')
        self.forward(D['X'])
        
        # Choose opitmizer
        optimizer = optim.Adam(self.parameters()) # here we are using the Adam optimizer, 
        # to optimize model.parameters, but what is there inside this attribute?
        
        
        # ========= Training loop
        if verbose:
            print('Starting training loop...')
        for ep in range(self.n_epochs):
            # Forward pass
            output = self.forward(D['X'])
            # compute loss on training data
            output_tr = self.alpha_t_transform(output, inverse=True)
            loss = self.loss(output_tr, D['Y'])
            
            # # Compute L1 regularization term
            # l1_regularization = 0.0
            # for param in model.parameters():
            #     l1_regularization += torch.norm(param, p=1)
            # # Add regularization term to the loss
            # loss += lambda_reg * l1_regularization
            
            # Compute loss on validation data.
            output_val = self.forward(D['X_val'])
            output_val_tr = self.alpha_t_transform(output_val, inverse=True)
            loss_val = self.loss(output_val_tr, D['Y_val'])
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train_loss_list.append(loss.item()) # Save the losses
            # test_loss_list.append(loss_test.item())
            if ep == 0:
                print(f'Epoch [{ep}/{self.n_epochs}], Training loss: {loss.item():.4f}, Validation loss: {loss_val.item():.4f}')
            if (ep + 1) % (self.n_epochs//10) == 0:
                print(f'Epoch [{ep+1}/{self.n_epochs}], Training loss: {loss.item():.4f}, Validation loss: {loss_val.item():.4f}')
        
        
        self.Y_pred = D["X"][:,2] # predict using the third column of the input data
        
        self.is_trained = True
        
        print('Model trained successfully')
        

    def predict(self, test_data_folder):
        '''
        The input will be the relative path of a folder with the following structure:
            
        test_data_folder_name
               ├── .DS_Store
               ├── chem_thermo_tran
               │   └── H2_9_42_0_SD.yaml
               ├── data
               │   ├── .DS_Store
               │   ├── C_grad_X_m-1_id000.dat
               │   ├── C_grad_Y_m-1_id000.dat
               │   ├── C_grad_Z_m-1_id000.dat
               │   ├── C_grad_m-1_id000.dat
               │   ├── C_id000.dat
               │   ├── P_Pa_id000.dat
               │   ├── RHO_kgm-3_id000.dat
               │   ├── T_K_id000.dat
               │   ├── UX_ms-1_id000.dat
               │   ├── UY_ms-1_id000.dat
               │   ├── UZ_ms-1_id000.dat
               │   └── YH2_id000.dat
               ├── grid
               │   ├── X_m.dat
               │   ├── Y_m.dat
               │   └── Z_m.dat
               └── info.json
               
        The predict function should be able to preprocess the data in a consistent way
        with respect to the training phase (same scaling, same input variables...).
        The predict() method is called from the ingestion code after the fit() method,
        so the model should be already trained when called.
        '''
        print(f'\Starting to process the data in {test_data_folder}...')
        with mute_print():
            field = ap.Field3D(test_data_folder)
        
        def build_tensor(field,
                         attributes
                         ):
            
            # field = self.data_dict[key]
            
            dim0 = product(field.shape)
            dim1 = len(attributes)
            
            X = np.zeros([dim0, dim1], dtype=np.float32)
            
            for i, attr in enumerate(attributes):
                X[:, i] = getattr(field, attr).value
            
            return X
        
        X = build_tensor(field, ['T', 'C', 'RHO', 'C_grad'])
        
        # Scale the matrix with the same parameters used previously
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        
        Y_pred = self.forward(X)
        Y_pred = self.alpha_t_transform(Y_pred, inverse=True)
        # IMPORTANT!!! Convert the output of the predict() function to a numpy array
        Y_pred = Y_pred.detach().numpy() 
        
        return Y_pred
    
    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        x = self.alpha_t_transform(x) # applying log transform to the output
        return x
    
    def loss(self, alpha_t, Y):
        # the columns in the matrix Y are (in order) : 
        # ['TAU_C_X', 'TAU_C_Y', 'TAU_C_Z', 'RHO', 'C_grad_X', 'C_grad_Y', 'C_grad_Z']
        # (see preprocess function)
        
        rho = Y[:, 3].unsqueeze(1)
        grad_C = Y[:, 4:6]
        
        Y_pred = rho * alpha_t * grad_C
        
        Y_true = Y[:, 0:2]
    
        MSE = torch.mean((Y_pred - Y_true).flatten() ** 2)
        
        return MSE
    
    # def alpha_t_transform(self, x, inverse=False):
        
    #     a = torch.log10(1+self.alpha_t_scaling*torch.abs(x))
        
    #     if not inverse:
    #         x_tr = torch.sign(x)*a
    #     else:
    #         x_tr = torch.sign(x)*((10**a)-1)/self.alpha_t_scaling
        
    #     return x_tr
    
    def alpha_t_transform(self, x, inverse=False):
        if not inverse:
            a = torch.log10(1 + self.alpha_t_scaling * torch.abs(x))
            x_tr = torch.sign(x) * a
        else:
            a = torch.abs(x)
            x_tr = torch.sign(x) * ((10 ** a) - 1) / self.alpha_t_scaling
        
        return x_tr

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.stats import norm
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

torch.manual_seed(42)
#np.random.seed(42)

# Will either train a new model, or load an existing one from a file
class MeasurementRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_d, out_d in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def set_up_model(X, Y, input_dim, hidden_dims, output_dim, model_dir="models", measurement_name=None,
                lr=1e-3, epochs=30, batch_size=64, device=None):
       
    """
    If a checkpoint exists at models/{measurement_name}.pt, load it.
    Otherwise, instantiate-and-train a new MeasurementRegressor and save it.
    Returns the trained model on CPU.
    """
    tensor_X = torch.tensor(X.values, dtype=torch.float32)
    tensor_Y = torch.tensor(Y.values, dtype=torch.float32)
    train_ds = TensorDataset(tensor_X, tensor_Y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(),)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{measurement_name}.pt")

    model = MeasurementRegressor(input_dim, hidden_dims, output_dim).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.cpu()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
            desc=f"[{measurement_name}] Epoch {ep+1}/{epochs}",
            leave=False)
        for X_batch, y_batch in pbar:
            Xb = X_batch.to(device, non_blocking=True)
            yb = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            ypred = model(Xb)
            loss = criterion(ypred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/len(train_loader.dataset):.4f}")
        print(f"[{measurement_name}] Epoch {ep+1}/{epochs}  loss = {epoch_loss/len(train_loader.dataset):.4f}")

    # save final weights
    torch.save(model.state_dict(), model_path)
    model.eval()
    return model.cpu()

def predict_errors(model, independent_df, dependent_df, device=None):     # here dependent data is the true values for the dependent variables, this can be used to compute errors
    """
    independent_df: pandas.DataFrame of shape [n_samples, n_indep_features]
    dependent_df:   pandas.DataFrame of shape [n_samples, n_dep_features]
    returns: np.ndarray of shape [n_samples, n_dep_features] = (y_true - y_pred)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    X = torch.tensor(independent_df.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_hat = model(X).cpu().numpy()

    Y_true = dependent_df.values
    return Y_true - Y_hat

def compute_confidences(sample_ids, errors, dep_cols, eps=1e-8):
    """
    errors: np.ndarray [n_samples, n_dep_features]
    For each dependent feature, fit a Gaussian on the TRAINING residuals
    (we'll assume `errors` here are from YOUR TRAINING SET).
    Then compute z-scores and map them to two-sided confidence:
    conf = 1 - |z| / max(|z|)   or   conf = norm.pdf(z)
    Returns: DataFrame indexed by sample_ids with one “confidence” per dep feature.
    """
    df_err = pd.DataFrame(errors, index=sample_ids, columns=dep_cols)
    confidences = pd.DataFrame(index=sample_ids)
    for col in df_err.columns:
        resid = df_err[col].values
        mu, sigma = resid.mean(), resid.std(ddof=0)
        sigma = max(sigma, eps)
        z = (resid - mu) / sigma
        confidences[col] = norm.pdf(z)      # add likelihoods

    return confidences


if __name__ == "__main__":
    # 1) load your full DataFrame
    df = pd.read_csv("refData_obf.csv", index_col="SampleId")
    traindf = df.copy()
    if traindf.isna().any().any():
        traindf = traindf.fillna(df.mean())  # takes mean of each column to fill NaNs

    testdf = df.copy()
    
    # 2) define your measurement groups
    measurements = [
        ("A_AL_CO", "A_B_CO", "A_CA_CO", "A_NI_RT", "A_CEC_CO", "A_CU_CO", "A_FE_CO", "A_K_CO", "A_MG_CO", "A_MN_CO", "A_NA_CO", "A_P_CO", "A_S_CO", "A_ZN_CO"),
        ("A_AL_M3", "A_B_M3", "A_CA_M3", "A_CU_M3", "A_FE_M3", "A_K_M3", "A_MG_M3", "A_MN_M3", "A_NA_M3", "A_P_M3", "A_S_M3", "A_ZN_M3"),
        ("A_AL_RT", "A_CA_RT", "A_CU_RT", "A_C_RT", "A_FE_RT", "A_K_RT", "A_MG_RT", "A_MN_RT", "A_NA_RT", "A_N_RT", "A_PB_RT", "A_P_RT", "A_SI_RT", "A_S_RT", "A_ZN_RT"),
        ("A_CLAY_MI", "A_SAND_MI", "A_SILT_MI"),
        ("A_CN_OF", "A_C_OF"),
        ("A_C_IF",),
        ("A_DENSITY_SA",),
        ("A_EC_WA", "A_PH_WA"),
        ("A_PH_KCL",),
    ]
    measurement_names = ["_CO", "_M3", "_RT", "_MI", "_OF", "_IF", "_SA", "_WA", "_KCL"]
    variables = list(df.columns)

    # 3) loop over each measurement, build/train or load, score
    all_confidences = []        # should be a list of dataframes, one per measurement
    for measurement, name in zip(measurements, measurement_names):
        independent_variables = [v for v in variables if v not in measurement]
        dependent_variables = list(measurement)

        X = traindf[independent_variables]
        Y = traindf[dependent_variables]

        model = set_up_model(X, 
                             Y, 
                             input_dim = X.shape[1],
                             hidden_dims = [500, 200],
                             output_dim = Y.shape[1],
                             measurement_name = name)

        errors = predict_errors(model, X, Y)            # shape [n, len(meas)]
        confidences = compute_confidences(df.index, errors, dependent_variables)         # DataFrame with N rows
        for col in dependent_variables:
            mask = testdf[col].isna()
            confidences.loc[mask, col] = 1.0        # set confidence to 1.0 for rows where the dependent variable is NaN
        measurement_confidences = confidences[dependent_variables].mean(axis=1)
        measurement_confidences.name = name    # name is "_CO", "_M3", etc.

        all_confidences.append(measurement_confidences)

    # 4) concatenate all measurement‐level confidences side by side
    raw_confidences = pd.concat(all_confidences, axis=1)
    print(raw_confidences.columns.tolist())
    final = raw_confidences[measurement_names]
    final.to_csv("measurement_outlier_confidences.csv")
    print("Finished! Outlier confidences saved to measurement_outlier_confidences.csv")

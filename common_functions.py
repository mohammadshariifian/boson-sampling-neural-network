# common_functions.py
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device = torch.device("cpu")

class QuantumKernelNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_state,
        modes: int,
        depth: int,
        *,
        block_a: int = 64,
        block_b: int = 64,
        chunk_subsets: int = 1 << 14,
        dtype_embed=torch.float32,
    ):
        super(QuantumKernelNN, self).__init__()

        self.device = torch.device("cpu")
        self.fc1 = nn.Linear(input_dim, output_dim).to(self.device)

        self.init_state = init_state
        self.modes = int(modes)
        self.depth = int(depth)

        # Blocked Gram params (RAM control)
        self.block_a = int(block_a)
        self.block_b = int(block_b)
        self.chunk_subsets = int(chunk_subsets)

        self.dtype_embed = dtype_embed

        # Photon occupation indices from init_state
        input_indices = []
        output_indices = []
        for mode, count in enumerate(init_state):
            input_indices.extend([mode] * count)
            output_indices.extend([mode] * count)

        self.register_buffer(
            "input_indices",
            torch.tensor(input_indices, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "output_indices",
            torch.tensor(output_indices, dtype=torch.long, device=self.device),
        )

    def angles_from_x(self, x: torch.Tensor):
        x = x.to(self.device)
        x_emb = torch.sigmoid(self.fc1(x)).to(self.dtype_embed)  # (N, output_dim)
        # z = self.fc1(x)
        # eps = 0.02  # try 0.01, 0.02, 0.05
        # x_emb = eps + (1 - 2*eps) * torch.sigmoid(z)

        N = x_emb.size(0)
        TBUargs = x_emb.view(N, -1, 2)                           # (N, num_TBU, 2)

        theta = (TBUargs[:, :, 0] * (torch.pi / 2)).to(torch.float32)
        phi   = (TBUargs[:, :, 1] * (2 * torch.pi)).to(torch.float32)

        return x_emb, TBUargs, theta, phi

    def unitaries_from_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build U(x) for each sample in x.
        returns U: (N, modes, modes) complex
        """
        x_emb, TBUargs, theta, phi = self.angles_from_x(x)
        N = x_emb.size(0)
        num_TBU = TBUargs.size(1)

        cos_theta = torch.cos(theta).to(torch.cfloat)           # (N, num_TBU)
        sin_theta = torch.sin(theta).to(torch.cfloat)           # (N, num_TBU)
        exp_phi   = torch.exp(1j * phi).to(torch.cfloat)        # (N, num_TBU)

        TBUunitaries = torch.zeros((N, num_TBU, 2, 2), dtype=torch.cfloat, device=self.device)
        TBUunitaries[:, :, 0, 0] = exp_phi * cos_theta
        TBUunitaries[:, :, 0, 1] = -sin_theta
        TBUunitaries[:, :, 1, 0] = exp_phi * sin_theta
        TBUunitaries[:, :, 1, 1] = cos_theta

        U = torch.eye(self.modes, dtype=torch.cfloat, device=self.device).repeat(N, 1, 1)
        N_before = 0
        for d in range(self.depth):
            U, N_before = apply_depth_layer_batch(d, self.modes, TBUunitaries, U, N_before)
        return U

    def forward(self, x: torch.Tensor):
        """
        x: (N, input_dim)
        returns:
          x_emb: (N, output_dim)
          quant_gram_prime: (N, N) float32
        """
        # Embed + build unitaries
        x_emb, _, _, _ = self.angles_from_x(x)   # keep embedding for training
        U = self.unitaries_from_x(x)             # (N, modes, modes) complex

        # photon-subspace indices
        idx = self.output_indices  # (nph,)

        # BLOCK-WISE Gram evaluation (RAM-safe)
        quant_gram_prime = photonic_gram_from_unitaries_blocked(
            U,
            idx,
            block_a=self.block_a,
            block_b=self.block_b,
            chunk_subsets=self.chunk_subsets,
        )

        return x_emb, quant_gram_prime



def get_rng_state():
    state = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state):
    random.setstate(state["py_random"])
    np.random.set_state(state["np_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])



def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  



def photonic_Gram_calculator(
    X, modes, depth, init_state,
    *,
    # xi: float = 1.0,
    block_a: int = 64,
    block_b: int = 64,
    chunk_subsets: int = 1 << 14,
):
    N = len(X)
    X = torch.stack(X) if isinstance(X, list) else X
    X = X.to(device)
    
    # Build per-sample interferometer unitaries U: (N, modes, modes)
    TBUargs = X.view(N, -1, 2)
    theta = TBUargs[:, :, 0] * (torch.pi / 2)
    phi   = TBUargs[:, :, 1] * (2 * torch.pi)

    cos_theta = torch.cos(theta).to(torch.cfloat)
    sin_theta = torch.sin(theta).to(torch.cfloat)
    exp_phi   = torch.exp(1j * phi)

    num_TBU = TBUargs.size(1)
    TBUunitaries = torch.zeros((N, num_TBU, 2, 2), dtype=torch.cfloat, device=device)
    TBUunitaries[:, :, 0, 0] = exp_phi * cos_theta
    TBUunitaries[:, :, 0, 1] = -sin_theta
    TBUunitaries[:, :, 1, 0] = exp_phi * sin_theta
    TBUunitaries[:, :, 1, 1] = cos_theta

    U = torch.eye(modes, dtype=torch.cfloat, device=device).repeat(N, 1, 1)
    N_before = 0
    for d in range(depth):
        U, N_before = apply_depth_layer_batch(d, modes, TBUunitaries, U, N_before)

    # Photon-subspace indices from init_state
    idx = torch.tensor(
        [mode for mode, count in enumerate(init_state) for _ in range(count)],
        dtype=torch.long,
        device=device
    )

    # Blocked Gram evaluation (RAM-safe)
    quant_gram_prime = photonic_gram_from_unitaries_blocked(
        U, idx, block_a=block_a, block_b=block_b, chunk_subsets=chunk_subsets
    )

    classic_gram_prime = 0
    return quant_gram_prime, classic_gram_prime



def photonic_gram_from_unitaries_blocked(
    U: torch.Tensor,
    idx: torch.Tensor,
    *,
    block_a: int = 64,
    block_b: int = 64,
    chunk_subsets: int = 1 << 14,
) -> torch.Tensor:
    """
    Compute photonic Gram matrix K_ij in blocks to control RAM.

    """
    device = U.device
    N = U.shape[0]
    idx = idx.to(device)

    # Output Gram matrix (float32)
    K = torch.zeros((N, N), device=device, dtype=torch.float32)

    # Only compute upper-triangle blocks; mirror to lower for symmetry.
    for a0 in range(0, N, block_a):
        a1 = min(a0 + block_a, N)
        UA_blk = U[a0:a1]  # (ba, m, m)
        UA_blk_dag = UA_blk.conj().transpose(-1, -2)  # (ba, m, m)

        for b0 in range(a0, N, block_b):
            b1 = min(b0 + block_b, N)
            UB_blk = U[b0:b1]  # (bb, m, m)

            # (ba, bb, m, m)
            Uab = torch.matmul(UA_blk_dag.unsqueeze(1), UB_blk.unsqueeze(0))

            # Restrict to photon subspace: (ba, bb, nph, nph)
            Uab_sub = Uab[:, :, idx[:, None], idx[None, :]]
            ba, bb, nph, _ = Uab_sub.shape

            # Flatten pairs: (ba*bb, nph, nph)
            Uflat = Uab_sub.reshape(ba * bb, nph, nph)

            # indistinguishable term: |perm(U)|^2
            perm = permanent_ryser(Uflat, chunk_subsets=chunk_subsets)  # complex
            P_indist = (perm.abs() ** 2).to(torch.float32)


            Pmix = P_indist

            Pblk = Pmix.reshape(ba, bb)  # (ba, bb)

            # Write into output (this keeps autograd connectivity)
            K[a0:a1, b0:b1] = Pblk
            if b0 != a0:
                K[b0:b1, a0:a1] = Pblk.transpose(0, 1)

            # Help Python free references sooner (esp. CPU RAM)
            del Uab, Uab_sub, Uflat, perm, P_indist, Pmix, Pblk

    # Exact self-overlap should be 1 (perm(I)=1)
    K.fill_diagonal_(1.0)
    return K



def make_mlp(input_dim: int, hidden_dims: list, output_dim: int,
             activation: str = "relu", final_sigmoid: bool = True) -> nn.Module:
    """
    Builds an MLP: input -> hidden_dims... -> output.
    If final_sigmoid=True, output is squashed to [0,1].
    """
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    Act = acts.get(activation.lower(), nn.ReLU)

    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(Act())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if final_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)



def pad_input(X_raw, modes, depth, init_state):
    if len(init_state) != modes:
        raise ValueError(f"Initial state length ({len(init_state)}) does not match the number of modes ({modes}).")
    
    if modes % 2 == 0:
        if depth % 2 == 0:
            num_circ_params = depth * modes - depth
        else:
            num_circ_params = depth * modes - (depth - 1)
    else:
        num_circ_params = depth * (modes - 1)

    length = len(X_raw[0])
    padded_X = []


    return padded_X, num_circ_params

def apply_depth_layer_batch(depth, modes, unitaries, U, N_before):
    N = U.size(0)
    device = U.device

    if modes % 2 == 0:
        if depth % 2 == 0 or modes == 2:
            num_blocks = modes // 2
            blocks = unitaries[:, N_before:N_before + num_blocks]  # (N, num_blocks, 2, 2)
            UBS = batch_block_diag(blocks, modes, device)
            U = torch.bmm(UBS, U)
            N_before += num_blocks
        else:
            num_blocks = (modes - 2) // 2
            blocks = unitaries[:, N_before:N_before + num_blocks]  # (N, num_blocks, 2, 2)
            UBS = batch_block_diag(blocks, modes, device, offset=1)
            U = torch.bmm(UBS, U)
            N_before += num_blocks
    else:
        num_blocks = modes // 2
        blocks = unitaries[:, N_before:N_before + num_blocks]
        if depth % 2 == 0:
            UBS = batch_block_diag(blocks, modes, device, add_1_at_end=True)
        else:
            UBS = batch_block_diag(blocks, modes, device, offset=1)
        U = torch.bmm(UBS, U)
        N_before += num_blocks

    return U, N_before

def batch_block_diag(blocks, modes, device, offset=0, add_1_at_end=False):
    N, num_blocks, _, _ = blocks.shape
    UBS = torch.eye(modes, dtype=torch.cfloat, device=device).repeat(N, 1, 1)

    for i in range(num_blocks):
        idx = offset + 2 * i
        UBS[:, idx:idx+2, idx:idx+2] = blocks[:, i]

    if add_1_at_end:
        UBS[:, -1, -1] = 1.0

    return UBS
  



def permanent(matrix, device):
    """
    Calculates the permanent of a square matrix.
    """
    matrix = matrix.to(device)  # Ensure the matrix is on the correct device
    
    # Base cases
    if matrix.shape[0] == 0:
        return torch.tensor(1.0, dtype=matrix.dtype, device=device)  # Return as a tensor on the same device
    
    if matrix.shape[0] == 1:
        return matrix[0, 0]
    
    result = torch.tensor(0.0, dtype=matrix.dtype, device=device)  # Ensure result is on the same device
    for i in range(matrix.shape[1]):
        submatrix = torch.cat((matrix[1:, :i], matrix[1:, i+1:]), dim=1)
        result += matrix[0, i] * permanent(submatrix, device)  # Recursive call with device
    
    return result




class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, mid_dim)   # Input → hidden
        self.fc2 = nn.Linear(mid_dim, output_dim)  # Hidden → output

    def forward(self, x):
        x = x.to(self.fc1.weight.device)           # Ensure correct device
        x = torch.sigmoid(self.fc1(x))             # Activation for hidden layer
        x_emb = torch.sigmoid(self.fc2(x))         # Activation for output
        return x_emb



def permanent_ryser(A: torch.Tensor, chunk_subsets: int = 1 << 14):
    """
    Batched Ryser permanent (works with complex A).
    A: (..., n, n) tensor (real or complex). Returns (...,) permanents.
    chunk_subsets: max number of subsets processed at once (tune for memory).
    """
    *batch_shape, n, m = A.shape
    assert n == m, "Matrix must be square"
    device = A.device
    A_flat = A.reshape(-1, n, n)   # [B, n, n]
    B = A_flat.shape[0]



    # Precompute integer masks for all non-empty subsets (1 .. 2^n - 1)
    num_subsets = (1 << n) - 1
    subsets = torch.arange(1, 1 << n, device=device, dtype=torch.long)   # [num_subsets]
    bits = torch.arange(n, device=device, dtype=torch.long)
    masks_int = ((subsets[:, None] >> bits[None, :]) & 1).to(torch.uint8)  # [num_subsets, n], uint8 saves memory

    perm = torch.zeros(B, dtype=A_flat.dtype, device=device)

    # chunk through subsets to limit memory
    chunk = int(chunk_subsets) if chunk_subsets and chunk_subsets > 0 else num_subsets
    for start in range(0, num_subsets, chunk):
        end = min(start + chunk, num_subsets)
        chunk_int = masks_int[start:end]                     # [C, n], dtype=uint8
        # Convert mask to same dtype as A_flat for einsum (will be real or complex with zero imag)
        chunk_mask = chunk_int.to(dtype=A_flat.dtype)        # [C, n], same dtype as A_flat (handles complex)
        # subset_sums: [B, C, n] where each entry is sum_{j in S} A[:, i, j]
        subset_sums = torch.einsum('bij,sj->bsi', A_flat, chunk_mask)
        # product over rows gives [B, C]
        prod_terms = subset_sums.prod(dim=2)
        # popcounts for signs (int)
        popcounts = chunk_int.sum(dim=1).to(torch.long)      # [C]
        sign_int = 1 - 2 * ((n - popcounts) & 1)             # +1 / -1 as int
        # cast sign to same dtype as prod_terms (complex-compatible)
        sign = sign_int.to(dtype=prod_terms.dtype, device=device)  # [C]
        perm = perm + (prod_terms * sign).sum(dim=1)

    return perm.reshape(*batch_shape)



class QuantumKernelLayeredNN(nn.Module):
    """
    Drop-in replacement that is GUARANTEED to be identical to QuantumKernelNN
    when nn_hidden_dims is empty (shallow case), by *delegating* to QuantumKernelNN.

    For deeper encoders (nn_hidden_dims non-empty), it uses an MLP encoder but
    keeps the unitary/Gram pipeline matched to QuantumKernelNN (dtype casts, device).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_state,
        modes: int,
        depth: int,
        nn_hidden_dims=None,
        activation: str = "relu",
        *,
        block_a: int = 64,
        block_b: int = 64,
        chunk_subsets: int = 1 << 14,
        dtype_embed=torch.float32,
    ):
        super().__init__()

        # Match QuantumKernelNN's device behavior exactly
        self.device = device

        if nn_hidden_dims is None:
            nn_hidden_dims = []
        self.nn_hidden_dims = list(nn_hidden_dims)

        self.init_state = init_state
        self.modes = int(modes)
        self.depth = int(depth)

        # Blocked Gram params (RAM control) — keep same names/behavior
        self.block_a = int(block_a)
        self.block_b = int(block_b)
        self.chunk_subsets = int(chunk_subsets)
        self.dtype_embed = dtype_embed

        # ------------------------------------------------------------
        # SHALLOW PATH: EXACT match to QuantumKernelNN
        # ------------------------------------------------------------
        self._is_shallow = (len(self.nn_hidden_dims) == 0)

        if self._is_shallow:
            # Delegate *everything* to QuantumKernelNN to guarantee identical behavior.
            self._qknn = QuantumKernelNN(
                input_dim=input_dim,
                output_dim=output_dim,
                init_state=init_state,
                modes=self.modes,
                depth=self.depth,
                block_a=self.block_a,
                block_b=self.block_b,
                chunk_subsets=self.chunk_subsets,
                dtype_embed=self.dtype_embed,
            )

            # Expose commonly-used attributes/buffers for compatibility
            self.fc1 = self._qknn.fc1
            self.input_indices = self._qknn.input_indices
            self.output_indices = self._qknn.output_indices
            return

        # ------------------------------------------------------------
        # DEEP PATH: MLP encoder, but match QKNN pipeline carefully
        # ------------------------------------------------------------
        self.encoder = make_mlp(
            input_dim=input_dim,
            hidden_dims=self.nn_hidden_dims,
            output_dim=output_dim,
            activation=activation,
            final_sigmoid=True,  # keep [0,1] like sigmoid(fc1)
        ).to(self.device)

        # Photon occupation indices from init_state (same as QuantumKernelNN)
        input_indices = []
        output_indices = []
        for mode, count in enumerate(init_state):
            input_indices.extend([mode] * count)
            output_indices.extend([mode] * count)

        self.register_buffer(
            "input_indices",
            torch.tensor(input_indices, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "output_indices",
            torch.tensor(output_indices, dtype=torch.long, device=self.device),
        )

    # -------------------------
    # Shallow delegation
    # -------------------------
    def forward(self, x: torch.Tensor):
        if self._is_shallow:
            return self._qknn(x)
        # deep path
        x_emb, _, _, _ = self.angles_from_x(x)
        U = self.unitaries_from_x(x)

        idx = self.output_indices
        quant_gram_prime = photonic_gram_from_unitaries_blocked(
            U,
            idx,
            block_a=self.block_a,
            block_b=self.block_b,
            chunk_subsets=self.chunk_subsets,
        )
        return x_emb, quant_gram_prime

    # -------------------------
    # Deep-path helpers
    # -------------------------
    def angles_from_x(self, x: torch.Tensor):
        x = x.to(self.device)
        x_emb = self.encoder(x).to(self.dtype_embed)   # keep dtype control like QuantumKernelNN
        N = x_emb.size(0)

        TBUargs = x_emb.view(N, -1, 2)
        theta = (TBUargs[:, :, 0] * (torch.pi / 2)).to(torch.float32)
        phi   = (TBUargs[:, :, 1] * (2 * torch.pi)).to(torch.float32)
        return x_emb, TBUargs, theta, phi

    def unitaries_from_x(self, x: torch.Tensor) -> torch.Tensor:
        # Match QuantumKernelNN casting EXACTLY
        x_emb, TBUargs, theta, phi = self.angles_from_x(x)
        N = x_emb.size(0)
        num_TBU = TBUargs.size(1)

        cos_theta = torch.cos(theta).to(torch.cfloat)
        sin_theta = torch.sin(theta).to(torch.cfloat)
        exp_phi   = torch.exp(1j * phi).to(torch.cfloat)

        TBUunitaries = torch.zeros((N, num_TBU, 2, 2), dtype=torch.cfloat, device=self.device)
        TBUunitaries[:, :, 0, 0] = exp_phi * cos_theta
        TBUunitaries[:, :, 0, 1] = -sin_theta
        TBUunitaries[:, :, 1, 0] = exp_phi * sin_theta
        TBUunitaries[:, :, 1, 1] = cos_theta

        U = torch.eye(self.modes, dtype=torch.cfloat, device=self.device).repeat(N, 1, 1)
        N_before = 0
        for d in range(self.depth):
            U, N_before = apply_depth_layer_batch(d, self.modes, TBUunitaries, U, N_before)
        return U
    



def convert_data(N,X_test, indexes_test, indexes_train, X_prime_train,model):
    ########################## test data conversion based on nn #############################
    X_test_tensor = torch.stack(X_test)
    model.eval()
    with torch.no_grad():  # Disables gradient computation
        # Pass through the model to get X_prime_test (the transformed test data)
        X_prime_test, quant_gram_test = model(X_test_tensor)
    # Now X_prime_test contains the transformed test data using the model with fixed weights

    ########################## new data and Gram construction #############################
    # X_prime = torch.cat((X_prime_train, X_prime_test), dim=0)
    X_prime = [None] * N  # Placeholder list
    
    for idx, x in zip(indexes_train, X_prime_train):
        X_prime[idx] = x

    for idx, x in zip(indexes_test, X_prime_test):
        X_prime[idx] = x

    # # Convert list to tensor if needed
    # X_prime = torch.stack(X_prime)  # Ensures it's a single tensor
    
    # # Check the final shape and output
    # print("X_prime shape:", X_prime.shape)
    return X_prime, X_prime_test
    
def zij_calc(num_train, y_train):
    y_train = y_train.view(-1, 1)  # Make y_train a column vector
    z_ij = (y_train == y_train.T).float()  # 1 if same class, 0 otherwise
    return z_ij



def SVM_acc_test(Gram_matrix, classifier, indexes_train, indexes_test, y_train, y_test):
    Gram_matrix = Gram_matrix.detach().cpu()
    Gram_np = Gram_matrix.numpy()
    Gram_matrix_train = Gram_np[np.ix_(indexes_train, indexes_train)]
    Gram_matrix_test = Gram_np[np.ix_(indexes_test, indexes_train)]
    # Gram_matrix_train = np.array([[Gram_matrix[i, j] for j in indexes_train] for i in indexes_train])
    # Gram_matrix_test = np.array([[Gram_matrix[i, j] for j in indexes_train] for i in indexes_test])
    classifier.fit(Gram_matrix_train, y_train)
    y_test_pred = classifier.predict(Gram_matrix_test)
    acc = accuracy_score(y_test, y_test_pred)

    return acc, y_test_pred

def SVM_acc_train(X_train, y_train, Gram_matrix, classifier):
    Gram_matrix = Gram_matrix.detach().cpu()
    X_train_train, X_train_test, y_train_train, y_train_test, indexes_train_train, indexes_train_test = train_test_split(X_train, y_train, range(len(X_train)),test_size=1/3)
    Gram_np = Gram_matrix.numpy()  # only do this once if Gram_matrix is a PyTorch tensor
    Gram_matrix_train_train = Gram_np[np.ix_(indexes_train_train, indexes_train_train)]
    Gram_matrix_train_test = Gram_np[np.ix_(indexes_train_test, indexes_train_train)]

    # Gram_matrix_train_train = np.array([[Gram_matrix[i, j] for j in indexes_train_train] for i in indexes_train_train])
    # Gram_matrix_train_test = np.array([[Gram_matrix[i, j] for j in indexes_train_train] for i in indexes_train_test])
    classifier.fit(Gram_matrix_train_train, y_train_train)
    y_train_test_pred = classifier.predict(Gram_matrix_train_test)
    acc_train = accuracy_score(y_train_test, y_train_test_pred)

    classifier.fit(Gram_matrix, y_train)
    y_train_pred = classifier.predict(Gram_matrix)
    return acc_train, y_train_pred


def get_leftmost_state(modes, ones_count):
    # Generate the state with ones at the leftmost positions
    state = [1] * ones_count + [0] * (modes - ones_count)
    name = ''.join(map(str, state))
    return name, state


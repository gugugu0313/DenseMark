import os
import torch
import torch.nn as nn
import numpy as np
import time
from torchvision.utils import save_image

def get_masked_bit_mode(mask_preds, post_info_bits, active_mask):
    """
    Majority vote over 32-bit binary vectors (avoids decimal overflow).
    Args:
        mask_preds:     [B, H, W]  bool or {0,1}
        post_info_bits: [B, 32, H, W]  {0,1}
        active_mask:    [B, H, W]  bool or {0,1}
    Returns:
        Tensor: [B, 32]  per-batch mode over valid pixels
    """
    combined_mask = mask_preds.int() & active_mask.int()
    B, C, H, W = post_info_bits.shape
    mode_bits = []

    for b in range(B):
        bits_batch = post_info_bits[b].permute(1, 2, 0).reshape(-1, C)
        valid_mask = combined_mask[b].flatten().bool()
        valid_bits = bits_batch[valid_mask]

        if len(valid_bits) == 0:
            mode_bit = torch.zeros(C, dtype=torch.int32, device=post_info_bits.device)
            mode_bits.append(mode_bit)
            continue

        weights = (1 << torch.arange(C, device=valid_bits.device, dtype=torch.int64)).unsqueeze(0)

        int_values = (valid_bits.to(torch.int64) * weights).sum(dim=1)
        unique_values, counts = torch.unique(int_values, return_counts=True)
        max_count_idx = torch.argmax(counts)
        mode_value = unique_values[max_count_idx]

        mode_bit = ((mode_value.unsqueeze(0) & weights) > 0).to(torch.int32).squeeze(0)
        mode_bits.append(mode_bit)

    mode_bits = torch.stack(mode_bits)
    return mode_bits

class DensePEG_LDPCBuilder:
    def __init__(self,
                 n=48,
                 k=32,
                 col_weights=None,  # per-column weights (irregular allowed)
                 seed=0,
                 device="cpu"):
        """
        n: codeword length
        k: information length
        col_weights: length-k list of column weights
        """
        assert n > k
        self.n = n
        self.k = k
        self.r = n - k
        self.device = device
        np.random.seed(seed)

        if col_weights is None:
            self.col_weights = [3]*k
        else:
            assert len(col_weights) == k
            self.col_weights = col_weights

        self.A = np.zeros((self.r, self.k), dtype=np.int8)
        self.H = None
        self.H_r = None

    # ------------------ Build systematic H ------------------
    def build(self, max_attempts_per_edge=10):
        for col in range(self.k):
            weight = self.col_weights[col]
            self._peg_connect_col(col, weight, max_attempts_per_edge)

        I = np.eye(self.r, dtype=np.int8)
        self.H = np.concatenate([self.A, I], axis=1)
        self.H_r = self.H.copy()

        self._sanity_check()

        self.H = torch.tensor(self.H, dtype=torch.int8, device=self.device)
        self.H_r = torch.tensor(self.H_r, dtype=torch.int8, device=self.device)
        return self.H, self.H_r

    # ------------------ PEG: connect one column ------------------
    def _peg_connect_col(self, col, weight, max_attempts=10):
        r = self.r
        row_degree = self.A.sum(axis=1)
        selected_rows = []

        for _ in range(weight):
            connected = False
            attempts = 0

            while not connected:
                min_deg = row_degree.min()
                candidates = np.where(row_degree == min_deg)[0]
                candidates = [row for row in candidates if row not in selected_rows]
                if len(candidates) == 0:
                    candidates = list(set(range(r)) - set(selected_rows))

                row = np.random.choice(candidates)
                self.A[row, col] = 1

                # Priority: avoid 4-cycles; else accept if column weight met; else force after max_attempts
                if not self._forms_4cycle(col):
                    connected = True
                    selected_rows.append(row)
                    row_degree[row] += 1
                elif len(selected_rows) + 1 >= weight:
                    connected = True
                    selected_rows.append(row)
                    row_degree[row] += 1
                else:
                    self.A[row, col] = 0
                    attempts += 1
                    if attempts >= max_attempts:
                        self.A[row, col] = 1
                        connected = True
                        selected_rows.append(row)
                        row_degree[row] += 1

    # ------------------ 4-cycle test for current column ------------------
    def _forms_4cycle(self, col):
        rows_col = np.where(self.A[:, col] == 1)[0]
        if len(rows_col) < 2:
            return False
        for other_col in range(self.A.shape[1]):
            if other_col == col:
                continue
            rows_other = np.where(self.A[:, other_col] == 1)[0]
            shared = np.intersect1d(rows_col, rows_other)
            if len(shared) >= 2:
                return True
        return False

    # ------------------ Column weights and rank ------------------
    def _sanity_check(self):
        col_weight = self.A.sum(axis=0)
        if np.any(col_weight < 2):
            raise ValueError("column weight < 2 (invalid)")
        rank = np.linalg.matrix_rank(np.concatenate([self.A, np.eye(self.r, dtype=int)], axis=1) % 2)
        if rank != self.r:
            raise ValueError(f"rank deficient: rank={rank}, r={self.r}")

    # ------------------ 4-cycle enumeration ------------------
    def detect_4cycles(self, H=None, verbose=True):
        if H is None:
            H = self.H.cpu().numpy()
        r, n = H.shape
        four_cycles = []
        for i in range(n):
            rows_i = np.where(H[:, i] == 1)[0]
            if len(rows_i) < 2:
                continue
            for j in range(i+1, n):
                rows_j = np.where(H[:, j] == 1)[0]
                if len(rows_j) < 2:
                    continue
                shared = np.intersect1d(rows_i, rows_j)
                if len(shared) >= 2:
                    four_cycles.append((i, j, shared.tolist()))
        if verbose:
            if four_cycles:
                print(f"❌ Found {len(four_cycles)} 4-cycles")
                for c in four_cycles[:5]:
                    print(f"  columns {c[0]} & {c[1]}, shared rows {c[2]}")
                if len(four_cycles) > 5:
                    print("  ...")
            else:
                print("✅ No 4-cycles detected")
        return four_cycles

    # ------------------ Save H / Hr ------------------
    def save(self, base_dir="output/ldpc"):
        # os.makedirs(base_dir, exist_ok=True)
        np.savetxt(f"{base_dir}/H_matrix_peg.csv", self.H.cpu().numpy(), delimiter=",", fmt="%d")
        np.savetxt(f"{base_dir}/Hr_matrix_peg.csv", self.H_r.cpu().numpy(), delimiter=",", fmt="%d")
        print("✅ H / Hr saved successfully.")


class StrictPEG_LDPCBuilder:
    """
    Strict PEG builder targeting girth >= 6 (v2).

    - All n columns via PEG (no trailing identity block).
    - Never accepts a new edge that creates a 4-cycle (unlike legacy DensePEG fallback).
    - Default column weights [3]*k + [2]*r (info dv=3, parity dv=2).
      Row-pair budget: k*C(3,2)+r*C(2,2)=3k+r <= C(m,2).
      For (n=48,k=32,m=16): 3*32+16=112 <= 120; uniform dv=3 would need 144>120 (impossible).
    - Among seeds, pick minimum row-weight variance.
    - Requires full GF(2) rank m.
    """

    def __init__(self,
                 n=48,
                 k=32,
                 col_weights=None,
                 max_seed=2000,
                 candidate_pool=30,
                 device="cpu"):
        assert n > k
        self.n = n
        self.k = k
        self.r = n - k
        self.device = device
        self.max_seed = max_seed
        self.candidate_pool = candidate_pool

        if col_weights is None:
            col_weights = [3] * k + [2] * self.r
        assert len(col_weights) == n, f"col_weights must have length n={n}, got {len(col_weights)}"

        total_pairs = sum(w * (w - 1) // 2 for w in col_weights)
        max_pairs = self.r * (self.r - 1) // 2
        assert total_pairs <= max_pairs, (
            f"Column weights infeasible: total row pairs {total_pairs} > C(m,2)={max_pairs}; "
            f"reduce column degrees for girth>=6."
        )
        self.col_weights = col_weights

        self.H = None
        self.H_r = None

    # ---------- One PEG attempt (strict girth>=6); None on failure ----------
    def _build_once(self, seed, max_col_retry=200):
        rng = np.random.default_rng(seed)
        m, n = self.r, self.n
        H = np.zeros((m, n), dtype=np.int8)
        row_deg = np.zeros(m, dtype=np.int64)

        def forms_4cycle(col, row):
            H[row, col] = 1
            rc = np.where(H[:, col] == 1)[0]
            if len(rc) < 2:
                H[row, col] = 0
                return False
            col_sums = H[rc].sum(axis=0)
            col_sums[col] = 0
            bad = (col_sums >= 2).any()
            H[row, col] = 0
            return bad

        for col in range(n):
            w = self.col_weights[col]
            success = False
            for _ in range(max_col_retry):
                H[:, col] = 0
                selected = []
                ok = True
                for _edge in range(w):
                    mask = np.ones(m, dtype=bool)
                    mask[selected] = False
                    unsel = np.where(mask)[0]
                    if len(unsel) == 0:
                        ok = False
                        break
                    deg = row_deg[unsel]
                    order = np.lexsort((rng.random(len(deg)), deg))
                    cand_rows = unsel[order]
                    placed = False
                    for r in cand_rows:
                        if not forms_4cycle(col, int(r)):
                            H[r, col] = 1
                            selected.append(int(r))
                            placed = True
                            break
                    if not placed:
                        ok = False
                        break
                if ok:
                    for r in selected:
                        row_deg[r] += 1
                    success = True
                    break
            if not success:
                return None
        return H

    @staticmethod
    def _gf2_rank(A):
        A = (A.copy() % 2).astype(np.int8)
        m, n = A.shape
        r = c = 0
        while r < m and c < n:
            p = None
            for i in range(r, m):
                if A[i, c] == 1:
                    p = i
                    break
            if p is None:
                c += 1
                continue
            if p != r:
                A[[r, p]] = A[[p, r]]
            for i in range(m):
                if i != r and A[i, c] == 1:
                    A[i] ^= A[r]
            r += 1
            c += 1
        return r

    @staticmethod
    def _count_4cycles(H):
        n = H.shape[1]
        cnt = 0
        cols = [np.where(H[:, j] == 1)[0] for j in range(n)]
        for i in range(n):
            if len(cols[i]) < 2:
                continue
            for j in range(i + 1, n):
                if len(np.intersect1d(cols[i], cols[j])) >= 2:
                    cnt += 1
        return cnt

    # ---------- Public API: build H ----------
    def build(self):
        found = []
        for seed in range(self.max_seed):
            H = self._build_once(seed)
            if H is None:
                continue
            if self._gf2_rank(H) != self.r:
                continue
            if self._count_4cycles(H) != 0:
                continue
            row_var = float(np.var(H.sum(axis=1)))
            found.append((seed, H, row_var))
            if len(found) >= self.candidate_pool:
                break

        if not found:
            raise RuntimeError(
                f"StrictPEG: no girth>=6 full-rank H found within {self.max_seed} seeds."
            )

        found.sort(key=lambda x: x[2])
        seed, H, _ = found[0]
        self.H = torch.tensor(H, dtype=torch.int8, device=self.device)
        self.H_r = self.H.clone()
        print(f"[StrictPEG] picked seed={seed}  row_wts={H.sum(axis=1).tolist()}  "
              f"4-cycles={self._count_4cycles(H)}  rank={self._gf2_rank(H)}")
        return self.H, self.H_r


class QCLDPCBuilder:
    """
    QC-LDPC parity-check builder; prototype matrix chosen from expansion factor Z.
    """

    def __init__(self, Z: int, device: str = 'cpu'):
        self.Z = Z
        self.device = device
        self.B = self._select_prototype(Z).to(device)
        self.mb, self.nb = self.B.shape
        self.m = self.mb * Z
        self.n = self.nb * Z

    @staticmethod
    def _select_prototype(Z: int) -> torch.Tensor:
        """Pick prototype base matrix B for given Z."""
        if Z == 4:
            # dv=3 short-code prototype
            B = torch.tensor([
                [ 0,  1,  2,  3, -1, -1,  0,  1, -1,  2, -1,  3],
                [ 1, -1,  3, -1,  0,  2, -1,  3,  1, -1,  0, -1],
                [-1,  2, -1,  0,  1, -1,  3, -1,  2,  1, -1,  0],
                [ 3, -1,  1, -1,  2,  0, -1,  2, -1, -1,  3,  1],
            ], dtype=torch.int64)
        elif Z == 8:
            # dv=2 feasible prototype
            B = torch.tensor([
                [ 0,  1,  2,  3,  4,  5],
                [ 3,  5,  7,  1,  6,  2],
            ], dtype=torch.int64)
        else:
            raise ValueError(f"No prototype matrix defined for Z={Z}")
        return B

    @staticmethod
    def _circulant_identity(Z: int, shift: int) -> torch.Tensor:
        I = torch.eye(Z, dtype=torch.int8)
        return torch.roll(I, shifts=shift, dims=1)

    def build_H(self) -> torch.Tensor:
        H_rows = []
        for i in range(self.mb):
            row_blocks = []
            for j in range(self.nb):
                shift = self.B[i,j].item()
                if shift < 0:
                    P = torch.zeros((self.Z,self.Z),dtype=torch.int8)
                else:
                    P = self._circulant_identity(self.Z, shift)
                row_blocks.append(P)
            H_rows.append(torch.cat(row_blocks,dim=1))
        H = torch.cat(H_rows,dim=0)
        return H.to(self.device, dtype=torch.int8)

    def summary(self):
        H = self.build_H()
        print(f"Z = {self.Z}")
        print(f"H shape: {H.shape}")
        print(f"Row weights: {H.sum(dim=1)}")
        print(f"Col weights: {H.sum(dim=0)}")



# ===========================
# 2. LDPC encoder
# ===========================


class LDPCEncoder:
    """
    Generic LDPC encoder (systematic H implied).
    Ensures c @ H.T % 2 == 0.
    """

    def __init__(self, H: torch.Tensor, device: str = "cpu", save_dir: str = None):
        """
        Args:
            H: parity-check matrix
            device: device for G
            save_dir: if set, dump G and Hr (RREF) for debugging
        """
        self.H = H.clone() % 2
        self.device = H.device
        self.m, self.n = self.H.shape
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.G = self.null_space_gf2(self.H)
        if self.save_dir is not None:
            np.savetxt(f"{self.save_dir}/G_matrix.csv", self.G.cpu().numpy(),
                       delimiter=",", fmt="%d")
        
    def gf2_rref(self, A: torch.Tensor):
        """
        Row-reduced echelon form over GF(2).
        Args:
            A: input GF(2) matrix.
        Returns:
            RREF matrix, list of pivot column indices.
        """
        A = A.clone() % 2
        m, n = A.shape
        pivot_cols = []
        row = 0
        for col in range(n):
            for r in range(row, m):
                if A[r, col] == 1:
                    A[[row, r]] = A[[r, row]]
                    break
            else:
                continue
            pivot_cols.append(col)
            for r in range(m):
                if r != row and A[r, col] == 1:
                    A[r] ^= A[row]
            row += 1
            if row == m:
                break
        return A, pivot_cols

    def null_space_gf2(self, check_matrix: torch.Tensor):
        """
        Null space of H over GF(2) -> generator matrix G.
        Args:
            check_matrix: parity-check H.
        Returns:
            Generator G (rows are basis codewords).
        """
        H_r, self.pivots = self.gf2_rref(check_matrix)
        if self.save_dir is not None:
            np.savetxt(f"{self.save_dir}/Hr_matrix.csv", H_r.cpu().numpy(),
                       delimiter=",", fmt="%d")
        free_cols = sorted(
            [j for j in range(check_matrix.shape[1]) if j not in self.pivots]
        )
        self.free_cols = free_cols  # cache for extract_info_bits
        basis = []
        for f in free_cols:
            v = torch.zeros(check_matrix.shape[1], dtype=torch.uint8)
            v[f] = 1
            for i, p in enumerate(self.pivots):
                v[p] = H_r[i, f] % 2
            basis.append(v)
        return torch.stack(basis).to(torch.int).to(self.device)

    def encode(self, msg: torch.Tensor):
        msg_float = msg.to(torch.float32)
        self_G_float = self.G.to(torch.float32)

        result = torch.matmul(msg_float, self_G_float) % 2

        return result.to(torch.int)


class FastLDPCDecoder(nn.Module):
    def __init__(self, H, max_iter=20, device='cpu'):
        """Fully vectorized LDPC decoder (SPA / min-sum)."""
        super().__init__()
        self.device = device
        self.H = H.float().to(device)
        self.r, self.n = self.H.shape
        self.max_iter = max_iter
        
        self.method = "spa" # "spa" or "min-sum"
        self.beta = 0.75
        self.llr_clip = 20.0
        self.gamma = 1.0001
        self.confidence_thr = 5.0  

        self.register_buffer('H_mask', self.H.unsqueeze(0).bool())
        self.register_buffer('H_t', self.H.t())
        self.register_buffer('idx_grid', torch.arange(self.n, device=device).view(1, 1, self.n))

    def hard_decision_and_check(self, llr):
        """Vectorized hard decision + syndrome check."""
        bits = (llr < 0).float()
        
        avg_conf = torch.mean(torch.abs(llr), dim=1)
        conf_mask = avg_conf >= self.confidence_thr

        syndromes = torch.remainder(torch.matmul(bits, self.H_t), 2.0)
        check_mask = torch.all(syndromes == 0, dim=1)

        is_valid = check_mask & conf_mask
        return bits, is_valid

    def forward(self, llr):
        batch_size = llr.shape[0]
        llr = llr.to(self.device).float()

        final_decoded_bits = torch.zeros(batch_size, self.n, device=self.device)
        final_llr_post = llr.clone()
        global_active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        bits, valid = self.hard_decision_and_check(llr)
        final_decoded_bits[valid] = bits[valid]
        final_llr_post[valid] = llr[valid]
        global_active_mask[valid] = False

        if not torch.any(global_active_mask):
            return final_llr_post, final_decoded_bits, ~global_active_mask

        active_idx = torch.nonzero(global_active_mask, as_tuple=True)[0]
        curr_batch_size = len(active_idx)

        curr_llr = llr[active_idx]
        curr_H_mask = self.H_mask.expand(curr_batch_size, -1, -1)
        C2V = torch.zeros(curr_batch_size, self.r, self.n, device=self.device)

        for it in range(self.max_iter):
            # 1) Variable-to-check messages
            total_C2V = C2V.sum(dim=1, keepdim=True)
            
            V2C = curr_llr.unsqueeze(1) + self.gamma * (total_C2V - C2V)
            V2C = torch.clamp(V2C, -self.llr_clip, self.llr_clip)
            
            V2C = torch.where(curr_H_mask, V2C, torch.zeros_like(V2C))

            # 2) Check-to-variable
            sign_V2C = torch.where(curr_H_mask, torch.sign(V2C), torch.ones_like(V2C))
            sign_V2C[sign_V2C == 0] = 1.0 
            total_sign = sign_V2C.prod(dim=2, keepdim=True)
            excluded_sign = total_sign * sign_V2C 

            if self.method == "min-sum":
                abs_V2C = torch.abs(V2C)
                abs_V2C = torch.where(curr_H_mask, abs_V2C, torch.tensor(float('inf'), device=self.device))
                
                min_vals, min_idx = torch.topk(abs_V2C, k=2, dim=2, largest=False)
                
                is_first_min = (self.idx_grid == min_idx[:, :, 0:1])
                excluded_min = torch.where(is_first_min, min_vals[:, :, 1:2], min_vals[:, :, 0:1])
                
                C2V = self.beta * excluded_sign * excluded_min

            else: 
                abs_V2C = torch.clamp(torch.abs(V2C), min=1e-6, max=self.llr_clip)
                
                phi_V2C = -torch.log(torch.tanh(abs_V2C / 2.0))
                phi_V2C = torch.where(curr_H_mask, phi_V2C, torch.zeros_like(phi_V2C))
                
                total_phi = phi_V2C.sum(dim=2, keepdim=True)
                excluded_phi = torch.clamp(total_phi - phi_V2C, min=1e-6)
                
                abs_C2V = -torch.log(torch.tanh(excluded_phi / 2.0))
                C2V = excluded_sign * abs_C2V

            C2V = torch.clamp(C2V, -self.llr_clip, self.llr_clip)
            C2V = torch.where(curr_H_mask, C2V, torch.zeros_like(C2V))

            LLR_post = curr_llr + C2V.sum(dim=1)
            curr_bits, curr_valid = self.hard_decision_and_check(LLR_post)

            if torch.any(curr_valid):
                newly_valid_original_idx = active_idx[curr_valid]
                
                final_decoded_bits[newly_valid_original_idx] = curr_bits[curr_valid]
                final_llr_post[newly_valid_original_idx] = LLR_post[curr_valid]
                global_active_mask[newly_valid_original_idx] = False

                still_active = ~curr_valid
                if not torch.any(still_active):
                    break

                active_idx = active_idx[still_active]
                curr_batch_size = len(active_idx)
                
                C2V = C2V[still_active]
                curr_llr = curr_llr[still_active]
                curr_H_mask = self.H_mask.expand(curr_batch_size, -1, -1)

        if torch.any(global_active_mask):
            LLR_post = curr_llr + C2V.sum(dim=1)
            final_llr_post[active_idx] = LLR_post
            curr_bits, _ = self.hard_decision_and_check(LLR_post)
            final_decoded_bits[active_idx] = curr_bits

        return final_llr_post, final_decoded_bits, ~global_active_mask
    
    
# ===========================
# 4. LDPC system wrapper
# ===========================
class LDPCSystem:
    """
    LDPC encode/decode wrapper.

    Loads H from ``matrix_dir``/ldpc_matrices.pt (girth>=6, no weight-1 columns expected).
    If missing, builds with StrictPEG_LDPCBuilder and saves.

    Args:
        max_iter: BP iterations
        device: torch device
        matrix_dir: directory for H/Hr/G; use "output/ldpc" for legacy matrices
    """

    def __init__(self,
                 max_iter: int = 30,
                 device: str = 'cpu',
                 matrix_dir: str = "checkpoints/ldpc"):
        self.matrix_dir = matrix_dir
        os.makedirs(self.matrix_dir, exist_ok=True)
        pt_path = os.path.join(self.matrix_dir, "ldpc_matrices.pt")

        if not os.path.exists(pt_path):
            print(f"[LDPCSystem] {pt_path} not found; building with StrictPEG_LDPCBuilder...")
            builder = StrictPEG_LDPCBuilder(n=48, k=32, device=device)
            H, H_r = builder.build()
            H = H.to(device)
            torch.save({"H": H, "H_r": H_r}, pt_path)
            np.savetxt(os.path.join(self.matrix_dir, "H_matrix.csv"),
                       H.cpu().numpy(), delimiter=",", fmt="%d")
            np.savetxt(os.path.join(self.matrix_dir, "Hr_matrix.csv"),
                       H_r.cpu().numpy(), delimiter=",", fmt="%d")
            print(f"[LDPCSystem] saved new matrices under {self.matrix_dir}/")
        else:
            data = torch.load(pt_path, map_location=device)
            H = data["H"].to(device)
            H_r = data["H_r"].to(device)

        self.encoder = LDPCEncoder(H, device, save_dir=self.matrix_dir)
        self.decoder = FastLDPCDecoder(H, max_iter=max_iter, device=device)

    def encode(self, b: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(b)

    def decode(self, llr: torch.Tensor) -> torch.Tensor:
        batch,bit,hight,width = llr.shape
        # print(f"bit_preds: {bit_preds[0,:,1,0].T}")
        llr_reshaped = llr.permute(0, 2, 3, 1) # [48, batch, H, W]
        llr_reshaped = - llr_reshaped.reshape(-1, bit)      # [B*H*W, 48]
        if llr.is_cuda: torch.cuda.synchronize()
        # t1 = time.time()
        LLR_post, decoded_bits, active_mask = self.decoder(llr_reshaped)
        if llr.is_cuda: torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"decode ms: {(t2 - t1)*1000:>6.2f}")
        post_info_bits = self.extract_info_bits(decoded_bits, self.encoder.pivots)
        active_mask = active_mask.float().reshape(batch, 1, hight, width)
        post_info_bits = post_info_bits.reshape(batch, hight,width, 32).permute(0,3,1,2) # [batch, 32, H, W]
        decoded_bits = decoded_bits.reshape(batch, hight, width, bit).permute(0, 3, 1, 2)  # [batch, 48, H, W]
        LLR_post = - LLR_post.reshape(batch, hight, width, bit).permute(0, 3, 1, 2)  # [batch, 48, H, W]

        return LLR_post, decoded_bits, post_info_bits, active_mask

    def extract_info_bits(self, decoded_bits: torch.Tensor, pivots: list) -> torch.Tensor:
        """
        Extract information bits (free columns) from decoded codewords.

        Must match ``free_cols`` order from LDPCEncoder.null_space_gf2 (use sorted list, not set iteration).
        """
        n = decoded_bits.shape[1]
        pivot_set = set(int(p) for p in pivots)
        self.free_cols = sorted([j for j in range(n) if j not in pivot_set])
        info_bits = decoded_bits[:, self.free_cols]
        return info_bits


def generate_llr_from_codeword(codeword: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
    """
    Synthetic LLRs from a codeword (toy channel).
    codeword: [..., n]
    noise_std: AWGN scale
    Returns LLR tensor (shape as expanded below).
    """
    n = codeword.shape[-1]
    codeword_expanded = codeword.expand(-1, n)

    noise = torch.randn_like(codeword_expanded, dtype=torch.float32) * noise_std

    llr = (1 - 2 * codeword_expanded.to(torch.float32)) + noise
    return llr.to(codeword.device)

import plotly.graph_objects as go
def save_3d_visualization(tensor, output_dir="output/ldpc"):
    """
    Save one Plotly 3D surface HTML per channel (e.g. 32 channels).
    Args:
        tensor: (1, 32, H, W) or compatible
        output_dir: base folder for HTML exports
    """
    output_dir = f"{output_dir}/3d_views"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tensor = tensor.squeeze(0).detach().cpu().numpy()  # (32, 256, 256)
    
    for i in range(tensor.shape[0]):
        fig = go.Figure()

        fig.add_trace(go.Surface(
            z=tensor[i, :, :],
            colorscale='Viridis',
            cmin=tensor.min(),
            cmax=tensor.max(),
            colorbar=dict(title='Value'),
        ))

        fig.update_layout(
            title=f"3D View of Channel {i}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        output_path = os.path.join(output_dir, f"channel_{i}.html")
        fig.write_html(output_path)

    print(f"Saved 3D view of channel to {output_path}")


def bits_to_rgb(bit_tensor):
    """
    Colorize bit patterns for visualization.
    bit_tensor: [B, C, H, W], C = number of bits (e.g. 32).
    """
    device = bit_tensor.device
    B, C, H, W = bit_tensor.shape
    
    powers = 2 ** torch.arange(C - 1, -1, -1, device=device, dtype=torch.long)
    powers = powers.view(1, C, 1, 1)
    
    values = (bit_tensor.long() * powers).sum(dim=1)
    
    flat_values = values.view(-1)
    uniq, inverse_idx = torch.unique(flat_values, return_inverse=True)
    
    num_colors = len(uniq)
    colors = torch.rand(num_colors, 3, device=device)
    
    color_mask = colors[inverse_idx]  # [B*H*W, 3]
    color_mask = color_mask.view(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
    
    return color_mask

if __name__ == "__main__":
    import torch
    device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(4)
    
    # Example 1: Z=4 path
    system_ldpc = LDPCSystem(max_iter=10, device=device)

    data = torch.load("output/ldpc/bit_preds_output.pt", map_location=device)
    msgs_encoded = data["msgs_encoded"]  # [batch, 32]
    bit_preds = data["bit_preds"]        # [batch, 48, H, W]
    codeword = data["codeword"]      # [batch, 48]
    mask_preds = data["mask_preds"]
    
    # save_3d_visualization(bit_preds, output_dir="output/ldpc")
    # batch,bit,hight,width = bit_preds.shape
    index =  (0, slice(None), 100, 100)
    print(f"bit_preds: {bit_preds[index].T}")
    print(f"codeword: {codeword[0,:]}")
    # bit_preds_reshaped = bit_preds.permute(0, 2, 3, 1) # [48, batch, H, W]
    # bit_preds_reshaped = - bit_preds_reshaped.reshape(-1, bit)      # [B*H*W, 48]
    
    input_llr = bit_preds[index].T
    input_llr_de = input_llr > 0
    diff_input_llr_de = input_llr_de.int() - codeword
    print("hard-decision error:", diff_input_llr_de)
    wrong_llr = diff_input_llr_de * bit_preds[index].T
    print("wrong_llr:", wrong_llr)
    # print(f"hard-decision error count: {diff_input_llr_de.abs().sum().item()}")
    
    # print(f"input_llr{input_llr}")
    LLR_post, decoded_bits, post_info_bits, active_mask = system_ldpc.decode(bit_preds)

    if (active_mask.float().mean() <= 0.001):
        active_mask = torch.ones_like(active_mask, dtype=torch.float32)
        
    pred_message = get_masked_bit_mode(mask_preds, post_info_bits, active_mask)
    pred_expanded = pred_message.unsqueeze(-1).unsqueeze(-1)  # [B, N_bits, 1, 1]
    bit_equal = (post_info_bits == pred_expanded)  # [B, N_bits, H, W]
    match_mask = bit_equal.all(dim=1)  # [B, H, W]
    save_image(match_mask.unsqueeze(1).float(), 'output/ldpc/match_mask.png')
    
    print(f"LLR_post: {LLR_post[index].T}")
    print(f"wrong_llr (scaled): {diff_input_llr_de * LLR_post[index].T}")
    # active_mask = active_mask.float().reshape(batch, 1, hight, width)
    # post_info_bits = post_info_bits.reshape(batch, hight,width, 32).permute(0,3,1,2) # [batch, 32, H, W]
    
    save_image(active_mask, 'output/ldpc/active_mask.png')

    diff = post_info_bits - msgs_encoded.unsqueeze(-1).unsqueeze(-1)

    post_bits_color = bits_to_rgb(post_info_bits)
    save_image(post_bits_color, 'output/ldpc/post_bits_color.png')

    rgb_diff = bits_to_rgb(diff)
    save_image(rgb_diff, 'output/ldpc/rgb_diff.png')
    
    # print(f"post_info_bits:\n {post_info_bits[256,:]}")
    # print(f"msgs_encoded:\n {msgs_encoded}")
    # print(f"diff[256,:]:\n {diff[256,:]}")
    # diff = diff.reshape(batch, hight,width, 32)
    # diff = diff.permute(0,3,1,2)  # [batch, 48, H, W]
    # print(f"diff[0,:,1,0]:\n {diff[0,:,1,0]}")
    diff_sum = torch.sum(torch.abs(diff), dim=1, keepdim=True)  # [batch, 1, H, W]

    
    diff_idx = diff_sum.clamp(max=6).squeeze(1).long()  # [1, 256, 256]

    color_lut = torch.tensor([
        [1,1,1],  # 0 white
        [0,1,0],  # 1 green
        [1,1,0],  # 2 yellow
        [1,0.5,0],# 3 orange
        [1,0,0],  # 4 red
        [0.5,0,0],# 5 dark red
        [0,0,0],  # 6+ black
    ], device=diff_sum.device).float()

    color_mask = color_lut[diff_idx]          # [1, 256, 256, 3]
    color_mask = color_mask.permute(0, 3, 1, 2)  # [1, 3, 256, 256]

    save_image(color_mask, 'output/ldpc/diff_mask_color.png')
    # diff_mask = (diff_sum == 0).float()  # [batch, 1, H, W]
    # print(f"diff_mask[0,:,1,0]:\n {diff_mask[0,:,1,0]}")
    # save_image(diff_mask, 'output/ldpc/diff_mask.png')
    # np.savetxt("output/ldpc/diff_mask.csv", diff_mask.squeeze(0).squeeze(0).cpu().numpy(), delimiter=",", fmt="%d")
    # print("encode/decode diff:", diff.abs().sum().item())


    info_bits = torch.randint(0, 2, (1,32), dtype=torch.int8, device=device)
    # print("info_bits:", info_bits)


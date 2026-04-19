import torch
import random

def process_ldpc_matrix(ldpc_matrix_path, device="cuda"):
    """
    Load an LDPC parity matrix and compute per-bit weights.

    Args:
        ldpc_matrix_path (str): Path to the LDPC matrix file.
        device (str): Compute device ("cuda" or "cpu").

    Returns:
        torch.Tensor: Normalized bit weights.
    """
    device = device if torch.cuda.is_available() else "cpu"

    # Load LDPC matrix
    data = torch.load(ldpc_matrix_path, map_location=device)
    H = data["H"].to(device)

    # ---------------- (1) Bit degrees ----------------
    bit_degrees = H.sum(dim=0).to(torch.int)  # column sum = bit node degree
    num_bits = H.shape[1]

    # ---------------- (2) Shared check-node groups ----------------
    check_bit_groups = []
    num_checks = H.shape[0]
    for chk in range(num_checks):
        bits_in_check = torch.nonzero(H[chk, :]).flatten().tolist()
        check_bit_groups.append(bits_in_check)

    # ---------------- (3) Detect 4-cycles (higher-risk bits) ----------------
    # Record whether each bit participates in a 4-cycle
    bit_in_4cycle = torch.zeros(num_bits, dtype=torch.int, device=device)
    for i in range(num_bits):
        for j in range(i+1, num_bits):
            common_checks = torch.nonzero(H[:, i] & H[:, j]).flatten().tolist()
            if len(common_checks) >= 2:
                bit_in_4cycle[i] = 1
                bit_in_4cycle[j] = 1

    # ---------------- (4) Within-group randomization (soft interleaving) ----------------
    randomized_groups = []
    for group in check_bit_groups:
        shuffled = group.copy()
        random.shuffle(shuffled)
        randomized_groups.append(shuffled)

    # ---------------- (5) Combined weights ----------------
    # Initial weights = degree
    bit_weights = bit_degrees

    # Extra weight if the bit lies in a 4-cycle
    bit_weights = bit_weights + bit_in_4cycle.float() * 0.3  # tunable 0.3

    # Small jitter within each check group
    for group in randomized_groups:
        for bit in group:
            bit_weights[bit] += random.uniform(0, 0.05)

    # Normalize to a smooth range
    bit_weights = torch.sigmoid(bit_weights.float())
    bit_weights = bit_weights + (1 - bit_weights.mean())

    return bit_weights

if __name__ == "__main__":
    ldpc_matrix_path = "output/ldpc/ldpc_matrices.pt"

    weights = process_ldpc_matrix(ldpc_matrix_path)
    print("Normalized bit weights:")
    print(weights)

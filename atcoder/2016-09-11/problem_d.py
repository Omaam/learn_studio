import numpy as np
from scipy.sparse import csr_matrix


def solver(H, W, N, a_vec=None, b_vec=None):

    # M = np.zeros((H, W), dtype=np.int32)
    M = csr_matrix((H, W))
    if a_vec is not None:
        for a, b in zip(a_vec, b_vec):
            M[a-1, b-1] = int(1)

    count = np.zeros(9)
    for j in range(9):
        for k in range(H-2):
            for n in range(W-2):
                num_panel = np.sum(M[k:k+3, n:n+3])
                if num_panel == j:
                    count[j] += 1
    count = count.astype(int)

    return list(count)


def main():
    try:
        H, W, N = map(int, input().split())
        ab = [map(int, input().split()) for _ in range(N)]
        a, b = [list(i) for i in zip(*ab)]
        print(solver(H, W, N, a, b))
    except ValueError:
        a, b = None, None
        print(solver(H, W, N, a, b))


if __name__ == "__main__":
    main()

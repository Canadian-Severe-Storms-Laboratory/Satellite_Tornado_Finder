#import torch
#import torch.nn as nn
#import onnxruntime as ort
import numpy as np
from time import perf_counter


# class DenseToCSR(nn.Module):
#     def forward(self, X, eps, minPts):
#
#         G = (X @ X.T)                          # (N, N)  -- inner products
#         A = G > eps
#
#         # core points: degree >= minPts
#         deg = A.to(torch.int64).sum(dim=1)                       # (N,)
#         core_mask = deg >= minPts.to(torch.int64)                # bool (N,)
#
#         # COO -> CSR
#         idx = torch.nonzero(A, as_tuple=False)                   # (nnz, 2), row-major
#         cols = idx[:, 1].to(torch.int64)                         # (nnz,)
#
#         # row_ptr = [0, cumsum(counts)]  where counts = sum over columns per row
#         row_ptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)], dim=0)  # (N+1,)
#
#         return row_ptr, cols, core_mask


if __name__ == '__main__':

    # A = np.random.randn(10000, 16).astype(np.float32)
    #
    # t = perf_counter()
    # G = A @ A.T
    # print(f"Elapsed: {perf_counter() - t:.3f} s")

    # D = 16
    # model = DenseToCSR().eval()
    # ex_X = torch.randn(3, D)
    # ex_eps = torch.tensor(0.5, dtype=torch.float32)
    # ex_minPts = torch.tensor(5, dtype=torch.int64)
    #
    # torch.onnx.export(
    #     model, (ex_X, ex_eps, ex_minPts),
    #     "dense_to_csr_dbscan.onnx",
    #     input_names=["X", "eps", "minPts"],
    #     output_names=["row_ptr", "col_idx", "core_mask"],
    #     opset_version=17,
    #     dynamic_axes={
    #         "X": {0: "N"},
    #         "row_ptr": {0: "N_plus_1"},
    #         "col_idx": {0: "nnz"},
    #         "core_mask": {0: "N"},
    #     },
    # )

    # model = SelfSimilarityThresholdFP16(threshold=0.0).eval()
    #
    # D = 16  # D fixed at export; N is dynamic
    # example = torch.randn(3, D)  # any N for tracing; model will cast internally
    #
    # torch.onnx.export(
    #     model,
    #     example,
    #     "aat_threshold_fp16.onnx",
    #     input_names=["A"],
    #     output_names=["Y"],
    #     opset_version=17,
    #     dynamic_axes={
    #         "A": {0: "N"},  # dynamic N on input
    #         "Y": {0: "N", 1: "N"},  # output is (N, N)
    #     },
    # )
    # print("Exported to aat_threshold_fp16.onnx")
    #
    # sess = ort.InferenceSession("aat_threshold_fp16.onnx", providers=['CUDAExecutionProvider'])
    #
    # N, D = 10000, 16
    # A = np.random.randn(N, D).astype(np.float32)  # fp32 is fine; model casts to fp16
    #
    # for i in range(10):
    #     t = perf_counter()
    #     (Y,) = sess.run(["Y"], {"A": A})
    #     print(f"Elapsed: {perf_counter() - t:.3f} s")


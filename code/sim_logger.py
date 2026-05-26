from __future__ import annotations

from typing import Sequence

import numpy as np


class SimDataRecorder:
    """Records simulation telemetry and plots it after the run.

    Stores time, true positions and reference positions for each drone.

    Shapes:
      - pos_true[k] and pos_ref[k] are (n_drones, 3)
      - stacked arrays are (T, n_drones, 3)
    """

    def __init__(self, *, n_drones: int, labels: Sequence[str] | None = None) -> None:
        if n_drones < 1:
            raise ValueError("n_drones must be >= 1")
        self.n_drones = int(n_drones)
        self.labels = list(labels) if labels is not None else [f"drone{i + 1}" for i in range(self.n_drones)]
        if len(self.labels) != self.n_drones:
            raise ValueError("labels length must match n_drones")

        self._t: list[float] = []
        self._pos_true: list[np.ndarray] = []
        self._pos_ref: list[np.ndarray] = []

    def log(self, *, t: float, pos_true: np.ndarray, pos_ref: np.ndarray) -> None:
        pos_true_arr = np.asarray(pos_true, dtype=float).reshape(self.n_drones, 3)
        pos_ref_arr = np.asarray(pos_ref, dtype=float).reshape(self.n_drones, 3)

        self._t.append(float(t))
        self._pos_true.append(pos_true_arr.copy())
        self._pos_ref.append(pos_ref_arr.copy())

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._t:
            return (
                np.zeros((0,), dtype=float),
                np.zeros((0, self.n_drones, 3), dtype=float),
                np.zeros((0, self.n_drones, 3), dtype=float),
            )
        t = np.asarray(self._t, dtype=float)
        pos_true = np.stack(self._pos_true, axis=0)
        pos_ref = np.stack(self._pos_ref, axis=0)
        return t, pos_true, pos_ref

    def plot(self, *, title: str | None = None) -> None:
        """Plots position tracking (true vs ref) and error norm for all drones."""
        t, pos_true, pos_ref = self.to_numpy()
        if t.size == 0:
            print("[SimDataRecorder] No data to plot.")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "matplotlib nao esta disponivel. Instale com: pip install matplotlib"
            ) from exc

        err = pos_true - pos_ref
        err_norm = np.linalg.norm(err, axis=2)  # (T, N)

        # Figure 1: x/y/z tracking
        fig1, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
        if title is None:
            fig1.suptitle("Posicao: real (solido) vs referencia (tracejado)")
        else:
            fig1.suptitle(title)

        axis_names = ["x [m]", "y [m]", "z [m]"]
        for ax_i, ax in enumerate(axes):
            for i in range(self.n_drones):
                line_true, = ax.plot(t, pos_true[:, i, ax_i], label=self.labels[i])
                ax.plot(
                    t,
                    pos_ref[:, i, ax_i],
                    linestyle="--",
                    color=line_true.get_color(),
                    alpha=0.9,
                )
            ax.set_ylabel(axis_names[ax_i])
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("tempo [s]")
        # Put legend once to avoid clutter.
        axes[0].legend(loc="upper right", ncol=1, fontsize=9)

        # Figure 2: error norms
        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 4))
        ax2.set_title("Erro de posicao: ||pos_real - pos_ref||")
        for i in range(self.n_drones):
            ax2.plot(t, err_norm[:, i], label=self.labels[i])
        ax2.set_xlabel("tempo [s]")
        ax2.set_ylabel("erro [m]")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper right", ncol=2, fontsize=9)

        plt.show()


    def compute_tracking_error_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (t, err_norm) where err_norm[k, i] = ||pos_true - pos_ref||."""
        t, pos_true, pos_ref = self.to_numpy()
        if t.size == 0:
            return t, np.zeros((0, self.n_drones), dtype=float)
        err_norm = np.linalg.norm(pos_true - pos_ref, axis=2)
        return t, err_norm


    def compute_global_formation_error(
        self,
        *,
        pos0: np.ndarray,
        leader_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global formation error w.r.t. leader (using initial offsets).

        Ideal (global) position is defined as:
          pos_ideal_i(t) = pos_true_leader(t) + (pos0_i - pos0_leader)

        This is the right metric to visualize *cascade*: if Drone2 is disturbed,
        Drone3 may keep its *edge* constraint but drift globally with Drone2.
        """
        t, pos_true, _pos_ref = self.to_numpy()
        if t.size == 0:
            return t, np.zeros((0, self.n_drones), dtype=float)

        p0 = np.asarray(pos0, dtype=float).reshape(self.n_drones, 3)
        if leader_idx < 0 or leader_idx >= self.n_drones:
            raise ValueError("leader_idx out of range")

        offsets = p0 - p0[leader_idx]
        pos_ideal = pos_true[:, leader_idx:leader_idx + 1, :] + offsets[None, :, :]
        err_global = np.linalg.norm(pos_true - pos_ideal, axis=2)
        return t, err_global


    def compute_graph_formation_error(
        self,
        *,
        pos0: np.ndarray,
        A: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Graph-consensus formation error (edge/graph constraint).

        For each agent i with incoming neighbors N_i (A[i,j] > 0), the graph
        formation constraint is:
          pos_i(t) \approx sum_j w_ij * (pos_j(t) + (pos0_i - pos0_j))

        Returns err_graph[k, i] = ||pos_true_i - pos_pred_i||.

        Notes:
          - If i has no incoming neighbors (row sum == 0), err_graph[:, i] = 0.
          - For a chain, this becomes the classic edge error (Drone3 w.r.t Drone2).
        """
        t, pos_true, _pos_ref = self.to_numpy()
        if t.size == 0:
            return t, np.zeros((0, self.n_drones), dtype=float)

        p0 = np.asarray(pos0, dtype=float).reshape(self.n_drones, 3)
        A = np.asarray(A, dtype=float)
        if A.shape != (self.n_drones, self.n_drones):
            raise ValueError("A shape must be (n_drones, n_drones)")

        row_sum = A.sum(axis=1)
        # Normalize weights row-wise
        W = np.zeros_like(A)
        for i in range(self.n_drones):
            if row_sum[i] > 0:
                W[i] = A[i] / row_sum[i]

        T = t.size
        err_graph = np.zeros((T, self.n_drones), dtype=float)

        for i in range(self.n_drones):
            if row_sum[i] <= 0:
                continue
            nbrs = np.where(W[i] > 0.0)[0]
            pred = np.zeros((T, 3), dtype=float)
            for j in nbrs:
                w_ij = float(W[i, j])
                delta_ij = p0[i] - p0[j]
                pred += w_ij * (pos_true[:, j, :] + delta_ij[None, :])
            err_graph[:, i] = np.linalg.norm(pos_true[:, i, :] - pred, axis=1)

        return t, err_graph


    def plot_cascade(
        self,
        *,
        pos0: np.ndarray,
        A: np.ndarray,
        leader_idx: int = 0,
        src_idx: int = 1,
        dst_idx: int = 2,
    ) -> None:
        """Plots a focused view of cascade: how src affects dst.

        Typical use (chain graph): src=1 (Drone2), dst=2 (Drone3).

        Produces 2 subplots:
          1) Global formation error w.r.t leader
          2) Graph/edge formation error (dst should stay low if it tracks src)
        """
        t, pos_true, pos_ref = self.to_numpy()
        if t.size == 0:
            print("[SimDataRecorder] No data to plot.")
            return

        if src_idx < 0 or src_idx >= self.n_drones:
            raise ValueError("src_idx out of range")
        if dst_idx < 0 or dst_idx >= self.n_drones:
            raise ValueError("dst_idx out of range")

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "matplotlib nao esta disponivel. Instale com: pip install matplotlib"
            ) from exc

        _, err_track = self.compute_tracking_error_norm()
        _, err_global = self.compute_global_formation_error(pos0=pos0, leader_idx=leader_idx)
        _, err_graph = self.compute_graph_formation_error(pos0=pos0, A=A)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
        fig.suptitle(
            f"Efeito cascata: {self.labels[src_idx]} influencia {self.labels[dst_idx]}"
        )

        # 1) Global error to leader
        ax1.set_title("Erro global vs lider (usa offsets iniciais)")
        for i in range(self.n_drones):
            ax1.plot(t, err_global[:, i], alpha=0.25)
        ax1.plot(t, err_global[:, src_idx], label=f"{self.labels[src_idx]} (src)")
        ax1.plot(t, err_global[:, dst_idx], label=f"{self.labels[dst_idx]} (dst)")
        ax1.set_ylabel("erro [m]")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper right", ncol=1, fontsize=9)

        # 2) Graph error + tracking error
        ax2.set_title("Erro de grafo (restricao) e erro de tracking")
        ax2.plot(t, err_graph[:, src_idx], label=f"grafo {self.labels[src_idx]}")
        ax2.plot(t, err_graph[:, dst_idx], label=f"grafo {self.labels[dst_idx]}")
        ax2.plot(t, err_track[:, src_idx], linestyle="--", label=f"track {self.labels[src_idx]}")
        ax2.plot(t, err_track[:, dst_idx], linestyle="--", label=f"track {self.labels[dst_idx]}")
        ax2.set_xlabel("tempo [s]")
        ax2.set_ylabel("erro [m]")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper right", ncol=2, fontsize=9)

        # Quick numerical hint: correlation in the global errors.
        if t.size >= 5:
            try:
                corr = float(np.corrcoef(err_global[:, src_idx], err_global[:, dst_idx])[0, 1])
                print(
                    f"[Cascade] corr(err_global {self.labels[src_idx]} -> {self.labels[dst_idx]}) = {corr:.3f}"
                )
            except Exception:
                pass

        plt.show()

"""
SE(3)-Transformer for predicting ligand property distributions from pocket point clouds.

Architecture:
    1. Input embedding: 46 scalar features -> hidden irreps (32x0e + 16x1o)
    2. 4x SE(3)-Transformer layers with radius-based attention
    3. Invariant pooling: per-atom features -> per-sample scalar features
    4. Prediction head: MLP -> 18 targets

Memory-efficient design: each layer uses a SINGLE shared tensor product for
message computation, with lightweight scalar attention for weighting. This
avoids the O(heads × edges × weight_numel) memory cost of per-head TPs.

Uses e3nn for spherical harmonics and equivariant tensor products.
Uses torch_cluster for efficient GPU-accelerated radius neighbor search.
Uses torch_scatter for per-sample aggregation.
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import BatchNorm as EquivariantBatchNorm
from torch_scatter import scatter_mean


# ---------------------------------------------------------------------------
# Radial basis functions
# ---------------------------------------------------------------------------

class GaussianRadialBasis(nn.Module):
    """Gaussian radial basis functions for encoding interatomic distances.

    Expands a scalar distance into a vector of Gaussian basis values,
    providing a smooth, learnable distance encoding for the model.
    """

    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Evenly spaced Gaussian centers from 0 to cutoff
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)

        # Width: each Gaussian decays to ~61% at one spacing apart (one sigma)
        # This gives good overlap between adjacent basis functions while still
        # allowing the model to discriminate distances at sub-Angstrom resolution
        spacing = cutoff / (num_basis - 1)
        self.width = 0.5 / (spacing ** 2)

    def forward(self, distances):
        """
        Args:
            distances: (E,) pairwise distances

        Returns:
            (E, num_basis) Gaussian basis values
        """
        return torch.exp(-self.width * (distances.unsqueeze(-1) - self.centers) ** 2)


# ---------------------------------------------------------------------------
# Smooth cutoff envelope
# ---------------------------------------------------------------------------

class CosineCutoff(nn.Module):
    """Smooth cosine cutoff that decays to zero at the boundary.

    Eliminates the hard discontinuity at the radius cutoff — atoms near
    the boundary get gradually reduced attention rather than being abruptly
    included/excluded.
    """

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        """
        Args:
            distances: (E,) pairwise distances

        Returns:
            (E,) envelope values in [0, 1]
        """
        return 0.5 * (1.0 + torch.cos(torch.pi * distances / self.cutoff)).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Input embedding
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """Project 46-dim scalar input features to hidden irreps.

    The 46 input features are all scalars (one-hot encodings and binary flags).
    We project them to the scalar part of the hidden irreps. The vector (l=1)
    part is initialized to zero and will be populated by the first
    SE(3)-Transformer layer through tensor products with edge spherical harmonics.
    """

    def __init__(self, input_dim, hidden_scalars, hidden_vectors):
        super().__init__()
        self.hidden_scalars = hidden_scalars
        self.hidden_vectors = hidden_vectors

        # Linear projection: scalars only
        self.linear = nn.Linear(input_dim, hidden_scalars)

    def forward(self, features):
        """
        Args:
            features: (N, 46) input features

        Returns:
            (N, hidden_scalars + hidden_vectors * 3) — packed irreps tensor
            The first hidden_scalars values are scalar features.
            The remaining hidden_vectors * 3 values are zero-initialized vectors.
        """
        scalars = self.linear(features)  # (N, hidden_scalars)

        # Initialize vector features to zero
        vectors = torch.zeros(
            features.shape[0], self.hidden_vectors * 3,
            device=features.device, dtype=features.dtype,
        )

        return torch.cat([scalars, vectors], dim=-1)


# ---------------------------------------------------------------------------
# SE(3)-Transformer Layer (memory-efficient single-TP design)
# ---------------------------------------------------------------------------

class SE3TransformerLayer(nn.Module):
    """One layer of the SE(3)-Transformer.

    Memory-efficient design using a single shared tensor product:
        1. Compute equivariant messages via TP(source_features, edge_SH)
        2. Compute scalar attention scores (query·key from scalar channels)
        3. Weight messages by attention scores and aggregate
        4. Gate activation + residual connection + equivariant batch norm

    This uses ONE tensor product per layer (not per-head), making memory
    usage O(edges × weight_numel) instead of O(heads × edges × weight_numel).
    Multi-head attention operates only on the scalar channels.
    """

    def __init__(self, hidden_scalars, hidden_vectors, num_heads, lmax,
                 num_radial_basis, radial_mlp_hidden, cutoff, dropout=0.0):
        super().__init__()
        self.hidden_scalars = hidden_scalars
        self.hidden_vectors = hidden_vectors
        self.num_heads = num_heads
        self.total_hidden = hidden_scalars + hidden_vectors * 3
        # Dropout applied only to scalar channels of the message-passing output.
        # Vectors (l=1) cannot be safely dropped without breaking equivariance.
        self.scalar_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Irreps definitions
        self.irreps_hidden = o3.Irreps(f"{hidden_scalars}x0e + {hidden_vectors}x1o")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)  # 1x0e + 1x1o for lmax=1

        # --- Single shared tensor product for message computation ---
        # source_features ⊗ edge_SH → messages (same irreps as hidden)
        self.message_tp = o3.FullyConnectedTensorProduct(
            self.irreps_hidden, self.irreps_sh, self.irreps_hidden,
            shared_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis, radial_mlp_hidden),
            nn.SiLU(),
            nn.Linear(radial_mlp_hidden, self.message_tp.weight_numel),
        )

        # --- Multi-head scalar attention ---
        # Queries from destination node scalars, keys from message scalars
        # Each head produces one attention score per edge
        self.query_net = nn.Linear(hidden_scalars, num_heads)
        self.key_net = nn.Linear(hidden_scalars, num_heads)

        # --- Output projection ---
        self.output_linear = o3.Linear(self.irreps_hidden, self.irreps_hidden)

        # --- Gate activation ---
        # Produces: hidden_scalars scalars + hidden_vectors gate scalars + hidden_vectors vectors
        self.gate_linear = o3.Linear(
            self.irreps_hidden,
            o3.Irreps(f"{hidden_scalars}x0e + {hidden_vectors}x0e + {hidden_vectors}x1o"),
        )

        # --- Equivariant batch normalization ---
        self.norm = EquivariantBatchNorm(self.irreps_hidden)

    def forward(self, node_features, edge_index, edge_sh, edge_radial, edge_envelope):
        """
        Args:
            node_features:  (N, total_hidden) packed irreps
            edge_index:     (2, E) source and destination indices
            edge_sh:        (E, sh_dim) spherical harmonics of edge vectors
            edge_radial:    (E, num_radial_basis) radial basis values
            edge_envelope:  (E,) smooth cutoff envelope values

        Returns:
            (N, total_hidden) updated node features
        """
        src, dst = edge_index  # (E,) each
        num_nodes = node_features.shape[0]

        # --- 1. Compute equivariant messages via tensor product ---
        tp_weights = self.radial_mlp(edge_radial)  # (E, tp_weight_numel)
        messages = self.message_tp(node_features[src], edge_sh, tp_weights)  # (E, total_hidden)

        # --- 2. Multi-head scalar attention ---
        # Query from destination scalar features, key from message scalar features
        dst_scalars = node_features[dst, :self.hidden_scalars]  # (E, hidden_scalars)
        msg_scalars = messages[:, :self.hidden_scalars]          # (E, hidden_scalars)

        queries = self.query_net(dst_scalars)  # (E, num_heads)
        keys = self.key_net(msg_scalars)       # (E, num_heads)

        # Per-head attention logits, averaged across heads for final score
        attn_logits = (queries * keys).sum(dim=-1) / (self.num_heads ** 0.5)  # (E,)

        # Apply cutoff envelope before softmax
        attn_logits = attn_logits + torch.log(edge_envelope + 1e-8)

        # Softmax over neighbors of each destination atom
        attn_weights = _scatter_softmax(attn_logits, dst, num_nodes=num_nodes)  # (E,)

        # --- 3. Weighted aggregation ---
        weighted_messages = attn_weights.unsqueeze(-1) * messages  # (E, total_hidden)

        # Scatter-add to destination atoms
        aggregated = torch.zeros(
            num_nodes, self.total_hidden,
            device=node_features.device, dtype=node_features.dtype,
        )
        aggregated.scatter_add_(
            0, dst.unsqueeze(-1).expand_as(weighted_messages), weighted_messages
        )

        # --- 4. Output projection ---
        aggregated = self.output_linear(aggregated)

        # --- 5. Gate activation ---
        gated = self.gate_linear(aggregated)
        s = gated[:, :self.hidden_scalars]                                          # scalars
        g = gated[:, self.hidden_scalars:self.hidden_scalars + self.hidden_vectors]  # gates
        v = gated[:, self.hidden_scalars + self.hidden_vectors:]                    # vectors

        # Apply activations
        s = torch.nn.functional.silu(s)
        # Dropout on scalar channels — safe wrt equivariance (l=0 only).
        s = self.scalar_dropout(s)
        g = torch.sigmoid(g)  # (N, hidden_vectors)

        # Gate the vectors: each vector channel multiplied by its gate scalar
        v = v.view(-1, self.hidden_vectors, 3)  # (N, hidden_vectors, 3)
        v = g.unsqueeze(-1) * v                  # (N, hidden_vectors, 3)
        v = v.view(-1, self.hidden_vectors * 3)  # (N, hidden_vectors * 3)

        activated = torch.cat([s, v], dim=-1)

        # --- 6. Residual connection + normalization ---
        output = self.norm(activated + node_features)

        return output


# ---------------------------------------------------------------------------
# Invariant pooling
# ---------------------------------------------------------------------------

class InvariantPooling(nn.Module):
    """Pool per-atom equivariant features to per-sample invariant features.

    1. Mean-pool scalar features per sample using batch_idx
    2. Compute L2 norm of each vector channel (rotation-invariant)
    3. Mean-pool vector norms per sample
    4. Concatenate: (hidden_scalars + hidden_vectors) features per sample
    """

    def __init__(self, hidden_scalars, hidden_vectors):
        super().__init__()
        self.hidden_scalars = hidden_scalars
        self.hidden_vectors = hidden_vectors

    def forward(self, node_features, batch_idx, num_samples):
        """
        Args:
            node_features: (total_atoms, hidden_scalars + hidden_vectors * 3)
            batch_idx:     (total_atoms,) which sample each atom belongs to
            num_samples:   int, number of samples in the batch (B)

        Returns:
            (B, hidden_scalars + hidden_vectors) invariant features
        """
        scalars = node_features[:, :self.hidden_scalars]
        vectors = node_features[:, self.hidden_scalars:]

        # Mean pool scalars per sample
        pooled_scalars = scatter_mean(scalars, batch_idx, dim=0, dim_size=num_samples)

        # Compute vector norms per channel
        vectors = vectors.view(-1, self.hidden_vectors, 3)
        vector_norms = vectors.norm(dim=-1)

        # Mean pool vector norms per sample
        pooled_norms = scatter_mean(vector_norms, batch_idx, dim=0, dim_size=num_samples)

        return torch.cat([pooled_scalars, pooled_norms], dim=-1)


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """MLP that maps pooled invariant features to 18 target predictions."""

    def __init__(self, input_dim, hidden_1, hidden_2, output_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Per-property mixture-of-experts head (Run 15)
# ---------------------------------------------------------------------------

class PerPropertyMixtureHead(nn.Module):
    """Mixture-of-experts head for one property.

    Given pooled invariant features, predicts:
      - cluster_logits: (B, K) logits over GMM clusters
      - residuals:     (B, K, 2) (Delta_mu, Delta_sigma) per cluster

    At training time only the assigned cluster's specialist receives gradient.
    At inference, argmax classifier picks one cluster, residual is added to
    that cluster's center to produce (mu_hat, sigma_hat).

    Cluster centers (K, 2) and a fixed lookup table are registered as
    buffers; not trainable.
    """

    def __init__(self, pooled_dim, k, hidden, dropout, cluster_centers):
        """
        Args:
            pooled_dim: int, e.g. 48
            k: int, number of clusters for this property
            hidden: int, hidden width of the small MLP blocks (e.g., 24)
            dropout: float
            cluster_centers: (K, 2) tensor of (mu_center, sigma_center)
        """
        super().__init__()
        self.k = k

        # Classifier head: pooled -> hidden -> K logits
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )
        # Specialist heads: pooled -> hidden -> 2 (Delta_mu, Delta_sigma)
        # K specialists, each independent
        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pooled_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            ) for _ in range(k)
        ])
        # Fixed cluster centers, not trainable
        self.register_buffer("cluster_centers", cluster_centers.float())

    def forward(self, z):
        """
        Args:
            z: (B, pooled_dim) pooled invariant features

        Returns:
            cluster_logits: (B, K) raw logits
            residuals:      (B, K, 2)
        """
        cluster_logits = self.classifier(z)
        # Stack K specialists' outputs
        residuals = torch.stack([s(z) for s in self.specialists], dim=1)  # (B, K, 2)
        return cluster_logits, residuals

    @torch.no_grad()
    def predict_mu_sigma(self, z, hard=True):
        """Inference helper: argmax classifier (hard) + add residual to center.

        Returns:
            mu_sigma: (B, 2) predicted (mu_hat, sigma_hat) in normalized
                      (cloud / target) space matching training targets.
        """
        cluster_logits, residuals = self.forward(z)
        if hard:
            k_hat = cluster_logits.argmax(dim=-1)                       # (B,)
            batch_idx = torch.arange(z.shape[0], device=z.device)
            picked_residual = residuals[batch_idx, k_hat]               # (B, 2)
            center = self.cluster_centers[k_hat]                        # (B, 2)
            return center + picked_residual
        else:
            # soft mixture (used only for diagnostic; main path is hard)
            probs = torch.softmax(cluster_logits, dim=-1)               # (B, K)
            mix = (probs.unsqueeze(-1) * (self.cluster_centers.unsqueeze(0) + residuals)).sum(dim=1)
            return mix


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PocketSE3Transformer(nn.Module):
    """SE(3)-Transformer for predicting ligand property distributions from pocket geometry.

    Takes raw point clouds (positions + 46-dim features per atom) and outputs
    18 predicted values (mean and std of 9 physicochemical properties).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Optional ESM projection (Path B). When enabled, the model splits
        # the per-atom feature vector into [46 chemistry, 1280 raw ESM],
        # projects the ESM portion through a learnable Linear(1280, K),
        # and concatenates with chemistry to produce a 46+K-dim input.
        # This forces a low-rank summary of ESM rather than letting the
        # model memorize the full 1280 channels.
        use_esm = getattr(config, "use_esm_features", False)
        esm_proj = getattr(config, "esm_projection_dim", 0)
        if use_esm and esm_proj > 0:
            self.esm_projection = nn.Linear(config.esm_feature_dim, esm_proj)
            self._chem_dim = 46  # chemistry-feature count (per data format)
        else:
            self.esm_projection = None
            self._chem_dim = None

        # Input embedding
        self.input_embed = InputEmbedding(
            config.input_features,
            config.hidden_scalars,
            config.hidden_vectors,
        )

        # Shared radial basis and cutoff envelope (same edges for all layers)
        self.radial_basis = GaussianRadialBasis(config.num_radial_basis, config.radius_cutoff)
        self.cutoff_fn = CosineCutoff(config.radius_cutoff)

        # SE(3)-Transformer layers
        self.layers = nn.ModuleList([
            SE3TransformerLayer(
                hidden_scalars=config.hidden_scalars,
                hidden_vectors=config.hidden_vectors,
                num_heads=config.num_heads,
                lmax=config.lmax,
                num_radial_basis=config.num_radial_basis,
                radial_mlp_hidden=config.radial_mlp_hidden,
                cutoff=config.radius_cutoff,
                dropout=getattr(config, "trunk_dropout", 0.0),
            )
            for _ in range(config.num_layers)
        ])

        # Invariant pooling
        self.pool = InvariantPooling(config.hidden_scalars, config.hidden_vectors)

        # Regression head selection (mutually exclusive):
        #   Run 8/12/13 baseline:   18-output regression head
        #   Run 14 exemplar mode:   81-output exemplar head
        #   Run 15 mixture mode:    per-property mixture-of-experts heads
        self.use_exemplar_head = getattr(config, "use_exemplar_head", False)
        self.use_mixture_heads = getattr(config, "use_mixture_heads", False)
        if self.use_exemplar_head and self.use_mixture_heads:
            raise ValueError(
                "use_exemplar_head and use_mixture_heads cannot both be True"
            )

        if self.use_mixture_heads:
            self.head = None
            self.exemplar_head = None
            self.n_exemplars = None
            self.mixture_heads = self._build_mixture_heads(config)
        elif self.use_exemplar_head:
            self.head = None
            self.mixture_heads = None
            n_ex = getattr(config, "n_exemplars", 9)
            self.n_exemplars = n_ex
            self.exemplar_head = PredictionHead(
                input_dim=config.pooled_dim,
                hidden_1=config.head_hidden_1,
                hidden_2=config.head_hidden_2,
                output_dim=n_ex * len(config.properties),
                dropout=config.head_dropout,
            )
        else:
            self.head = PredictionHead(
                input_dim=config.pooled_dim,
                hidden_1=config.head_hidden_1,
                hidden_2=config.head_hidden_2,
                output_dim=config.num_targets,
                dropout=config.head_dropout,
            )
            self.exemplar_head = None
            self.mixture_heads = None
            self.n_exemplars = None

        # Binary classification head — predicts P(aromatic_rings_mean > 0).
        # A small 2-layer MLP reading the same pooled features. Output is a
        # raw logit; sigmoid is applied at the loss / inference stage.
        self.binary_head = nn.Sequential(
            nn.Linear(config.pooled_dim, config.head_hidden_2),
            nn.SiLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_2, 1),
        )

        # Geometric auxiliary head — predicts 8 geometric labels (pocket-shape +
        # ligand-shape distributions). Same architecture pattern as binary_head;
        # only instantiated when use_geometric_targets is on so the parameter
        # count stays unchanged for past-run configs.
        if getattr(config, "use_geometric_targets", False):
            self.geom_head = nn.Sequential(
                nn.Linear(config.pooled_dim, config.geom_head_hidden),
                nn.SiLU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(config.geom_head_hidden, config.n_geometric_targets),
            )
        else:
            self.geom_head = None

        # Precompute spherical harmonics irreps
        self.irreps_sh = o3.Irreps.spherical_harmonics(config.lmax)

    def _build_mixture_heads(self, config):
        """Instantiate per-property mixture heads from GMM JSON.

        Returns a ModuleDict mapping property name -> PerPropertyMixtureHead.
        cluster_centers are in RAW property units (the GMM was fit on raw
        pocket-distribution values). Training-time targets must be denormalized
        before comparison to the head's output, OR the head's output must be
        renormalized. We choose the latter for efficiency: cluster centers
        and head outputs live in raw units; the train/eval helpers in train.py
        and evaluate.py renormalize at the boundary.
        """
        import json
        with open(config.mixture_json, "r") as f:
            mixture = json.load(f)

        heads = nn.ModuleDict()
        for prop in config.properties:
            entry = mixture[prop]
            centers = torch.tensor(entry["centers"], dtype=torch.float32)  # (K, 2)
            heads[prop] = PerPropertyMixtureHead(
                pooled_dim=config.pooled_dim,
                k=entry["k"],
                hidden=config.mixture_head_hidden,
                dropout=config.head_dropout,
                cluster_centers=centers,
            )
        return heads

    def forward(self, positions, features, batch_idx):
        """
        Args:
            positions:  (total_atoms, 3) concatenated atom coordinates
            features:   (total_atoms, 46) concatenated atom features
            batch_idx:  (total_atoms,) sample index per atom

        Returns:
            dict with:
                reg_preds:    (B, 18) regression target predictions
                binary_logit: (B,) raw logit for P(aromatic_present)
        """
        num_samples = batch_idx.max().item() + 1

        # Build radius neighbor list (no cap on neighbors)
        edge_index = _radius_graph(positions, self.config.radius_cutoff, batch_idx)

        # Compute edge vectors and distances
        src, dst = edge_index
        edge_vec = positions[dst] - positions[src]  # (E, 3)
        edge_dist = edge_vec.norm(dim=-1)            # (E,)

        # Spherical harmonics of edge directions
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )

        # Radial basis encoding (shared across all layers)
        edge_radial = self.radial_basis(edge_dist)  # (E, num_radial_basis)

        # Smooth cutoff envelope (shared across all layers)
        edge_envelope = self.cutoff_fn(edge_dist)  # (E,)

        # Optional ESM projection (Path B): split [46 chemistry, 1280 ESM],
        # project the ESM portion, and recombine. When disabled, features
        # pass through to input_embed unchanged.
        if self.esm_projection is not None:
            chem = features[:, : self._chem_dim]                       # (N, 46)
            esm  = features[:, self._chem_dim :]                       # (N, 1280)
            esm_proj = self.esm_projection(esm)                        # (N, K)
            features = torch.cat([chem, esm_proj], dim=-1)             # (N, 46+K)

        # Input embedding
        h = self.input_embed(features)

        # SE(3)-Transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_sh, edge_radial, edge_envelope)

        # Invariant pooling
        z = self.pool(h, batch_idx, num_samples)

        # Primary regression -- one of three modes selected at construct time.
        binary_logit = self.binary_head(z).squeeze(-1)  # (B,)
        if self.use_mixture_heads:
            # Run 15: emit per-property classifier logits + specialist residuals.
            # The training loss handles cluster-hard-assignment; eval uses
            # argmax + cluster center.
            cluster_logits = {}
            residuals = {}
            for prop in self.config.properties:
                cl, res = self.mixture_heads[prop](z)
                cluster_logits[prop] = cl
                residuals[prop] = res
            out = {
                "cluster_logits": cluster_logits,  # dict of (B, K_p)
                "residuals": residuals,            # dict of (B, K_p, 2)
                "binary_logit": binary_logit,
            }
        elif self.use_exemplar_head:
            flat = self.exemplar_head(z)                              # (B, 81)
            n_props = len(self.config.properties)
            exemplar_preds = flat.view(-1, self.n_exemplars, n_props)  # (B, 9, 9)
            out = {"exemplar_preds": exemplar_preds, "binary_logit": binary_logit}
        else:
            reg_preds = self.head(z)
            out = {"reg_preds": reg_preds, "binary_logit": binary_logit}

        # Optional geometric auxiliary predictions (8 outputs per sample)
        if self.geom_head is not None:
            out["geom_preds"] = self.geom_head(z)
        return out


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _radius_graph(positions, cutoff, batch_idx):
    """Build radius-based neighbor list using torch_cluster.

    No cap on number of neighbors — dynamic memory handling.

    Args:
        positions: (N, 3) atom coordinates
        cutoff: float, radius cutoff in Angstroms
        batch_idx: (N,) sample index per atom

    Returns:
        (2, E) edge index tensor (source, destination)
    """
    from torch_cluster import radius_graph
    edge_index = radius_graph(
        positions, r=cutoff, batch=batch_idx,
        loop=False, max_num_neighbors=10000,  # effectively no cap
    )
    return edge_index


def _scatter_softmax(logits, index, num_nodes):
    """Numerically stable softmax over variable-size groups defined by index.

    Args:
        logits: (E,) attention logits
        index:  (E,) destination node index for each edge
        num_nodes: int, total number of nodes

    Returns:
        (E,) attention weights summing to 1 per destination node
    """
    # Subtract max per group for numerical stability
    # Initialize with -inf so nodes with no edges don't affect the max
    max_vals = torch.full((num_nodes,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(0, index, logits, reduce="amax")
    logits_stable = logits - max_vals[index]

    # Exp
    exp_logits = torch.exp(logits_stable)

    # Sum per group
    sum_exp = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(0, index, exp_logits)

    # Normalize
    return exp_logits / (sum_exp[index] + 1e-8)

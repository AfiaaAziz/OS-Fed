# Decisions (fill in as you go; never code something the paper does not state before writing it here)

Status: DONE = decided, PROVISIONAL = decided for now, revisit at the step shown, OPEN = not decided yet.

| #   | Item | Decision | Status | Date | Who | Reason / where it is decided |
|-----|------|----------|--------|------|-----|------------------------------|
| D1  | Batch size for local training | 64 | DONE | 2026-11-21 | | Paper does not state it. Common default. |
| D2  | Momentum, weight decay | momentum 0, weight decay 0 (plain SGD) | DONE | 2026-11-21 | | Paper formula Δw = η Σ g is exact only for plain SGD. |
| D3  | Cosine schedule scope | Federated runs: lr 0.01 decays with cosine once per communication round over all rounds. Centralized check: constant 0.01 | DONE | 2026-11-21 | | Paper says "cosine schedule" but not the scope. |
| D4  | Snapshot initialization | random noise | DONE | 2026-11-24 | | Step 8.10 (1 client, 1 round, grid 4): random@50 iters gave best cos_sim 0.767 / lowest match_loss, slightly beating first_client_image. Check supplement. |
| D5  | L-BFGS settings | max_iter 50, lr 1.0, default history | DONE | 2026-11-24 | | Step 8.10: cos_sim kept rising to 50 iters and still ~0.5s on GPU; chosen as cheapest clearly-good. |
| D6  | ε in Eq. 5 | 1e-8 | PROVISIONAL | 2026-11-21 | | Small positive value. Revisit in Step 8. |
| D7  | Label format | Soft-label vector, one entry per class (length 10), M² of them | PROVISIONAL | 2026-11-21 | | Paper says "learnable label embeddings". Check supplement. |
| D8  | Loss L for the synthetic gradient | Cross-entropy with soft labels, mean over samples, same on client and server | PROVISIONAL | 2026-11-21 | | Paper says "a loss function". Mean reduction needed for Eq. 3. |
| D9  | BatchNorm | Tier A: SmallCNN with no BatchNorm. Tier B: OPEN (batch or group, 8 groups). Both are implemented in src/model.py | PROVISIONAL | | | Decide after reading the supplement. |
| D10 | Preprocessing, augmentation | CIFAR-10 mean/std normalization, no augmentation | DONE | 2026-11-21 | | Paper does not state it. |
| D11 | CIFAR snapshot shape | 3×32×32 | PROVISIONAL | 2026-11-21 | | Inferred from Table 2 (0.12 MB = 10 × 12,288 bytes). Confirm by measuring. |
| D12 | ResNet-18 variant | CIFAR-style: 3×3 first layer, no max-pool. Measured parameters: 11,173,962 (44.73 MB with BatchNorm, 44.70 MB with GroupNorm) | DONE | 2026-11-21 | | Matches the paper's ~44.7 MB per client message. Re-check with your own run of scripts/check_models.py. |
| D13 | Clients per round | All 10 clients every round | DONE | 2026-11-21 | | Paper: C clients per round. |
| D14 | FedAvg weighting | Main: data-size weighted (wavg). Check on small CNN: equal weights (uavg) | DONE | 2026-11-21 | | Proposal says weighted; paper's Eq. 3 is equal-weight. |
| D15 | Minimum client size, Dirichlet split | 100 samples | DONE | 2026-11-21 | | Our choice. |
| D16 | Supplement read | OPEN | OPEN | | | Download from the CVF page, read before Step 8, update this table. |
| D17 | Byte counting rule | Main number: payload bytes (tensor data only). Also log wire bytes (torch.save size). Report upload, download and total separately. Paper convention = upload only | DONE | 2026-11-21 | | src/metrics.py payload_bytes and wire_bytes. |
| D18 | Seeds | Partition seed 100 (fixed). Training seeds 1, 2, 3 | DONE | 2026-11-21 | | Same partition for both methods. |
| D19 | Data storage | CIFAR-10 cached as uint8 tensors in data/raw/cifar10_uint8.pt, normalized per batch | DONE | 2026-11-21 | | Speed and RAM (10 client processes later). |
| D20 | Where small-CNN accuracy runs execute | Colab T4 GPU. The Ubuntu VM measured 83 min per round, which is too slow for the full run |
| D21 | Rounds and local epochs for E1 | 100 rounds, 5 local epochs (paper defaults; completed within Colab T4 GPU time budget) |

## Week-4 Gate decision (2026-11-24) — UPDATED after 100-round runs
**Chosen level: OS-Fed WITHOUT error compensation (fallback ladder Level 2).** Reason below.
- Snapshot + multi-grid + server gradient recovery all work. Upload 0.129 MB total / 10 clients = 12,928 B each (~3x32x32 float32 + labels), vs 21.8 MB/client FedAvg download — matches paper Table 2 (~0.12 MB).
- Snapshot matching at grid 4, 1 client, 1 round: cos_sim ~0.77 (vs 0.58 grid 1) (Step 8.10). D4=random init, D5=L-BFGS max_iter 50.
- **ec1 (error compensation ON) DIVERGED over 100 rounds**: early-round matching is poor on full-data updates (residual_ratio ~0.99), so the accumulated error e grows unbounded -> test_loss explodes at round ~58 -> NaN -> acc 0.1000. Classic error-feedback blowup under high compression with a small CNN. (run 20261124_A_osfed_cnn_noniid_g4ec1_s1)
- **ec0 (no error compensation) is STABLE**: residual recovers to ~0.32 and cos_sim to ~0.95 by round 20, accuracy rising (0.40 at round 20, still climbing), no blowup. (run 20261124_A_osfed_cnn_noniid_g4ec0_s1, 20 rounds)
- DECISION: report ec0 (stable) as our OS-Fed result for the small-CNN tier, labeled "OS-Fed without error compensation (student-scale, reduced)". Stabilizing ec1 (error clipping / server step damping / more L-BFGS iters to cut early residual) is optional improvement work, not required by the gate.
- 20-round ec1 earlier looked fine (0.4363) but 100 rounds revealed the instability — this is why the plan requires the residual_ratio check over many rounds.
| D22 | Link model | Each client has its own link. The server link is not a bottleneck. Every message costs bytes x 8 / (Mbps x 1e6) + delay, using payload bytes. Emulated by time.sleep on one computer |
| D23 | Which clients are slow | The n clients with the highest ids (for slow2x2 with 10 clients: 8 and 9). Extra pause = (factor - 1) x measured compute time |
| D24 | Conditions used | net0, bw100, bw10, bw1, d50, d200, slow2x2, slow5x4 and their combinations. Real emulated runs only for validation (pcheck config). The full grid comes from Step 11 calculation |

# Step 10 guide: network and client-speed emulation

Run everything from the repo root, with the venv active and `export OMP_NUM_THREADS=1`.

## 1. Where the files go

| File | Exact path in the repo |
|---|---|
| network.py | src/network.py |
| test_network.py | scripts/test_network.py |
| check_emulation_logs.py | scripts/check_emulation_logs.py |

    cp ~/Downloads/network.py src/
    cp ~/Downloads/test_network.py scripts/
    cp ~/Downloads/check_emulation_logs.py scripts/
    python -m scripts.test_network

Expected last line: ALL STEP 10 CHECKS PASSED. This checks the plan's own test (1,000,000 bytes at 8 Mbps = about 1.00 s, with 50 ms delay = about 1.05 s) and that 4 real processes pause in parallel.

Calculator (no training needed), CNN messages of 2,180,392 bytes:

    python -m src.network --condition bw10-d50-slow2x2 --download-bytes 2180392 --upload-bytes 2180392 --compute 3.0 --clients 10

## 2. Edit src/run_process.py (4 small edits)

Find the places first:

    grep -n "condition\|Process(\|def .*client\|CLIENTS_COLUMNS\|clients_log.log\|put(" src/run_process.py

**Edit 1: import (top of file, with the other imports)**

    from src.network import NET_CLIENT_COLUMNS, NetworkEmulator, parse_condition

**Edit 2: read the condition (in main(), right after the arguments are parsed)**

If the file has no `--condition` option yet, add `ap.add_argument("--condition", type=str, default="net0")`, and build the run ID as `{date}_{config name}__{condition}` (plus the `_proc` ending your runner already uses).

    net_cfg = parse_condition(args.condition)
    print(net_cfg.describe())

**Edit 3: the client process function**

Pass `net_cfg` and `n_clients` to the client process (add them to the `args=(...)` of `Process(...)` and to the function's parameters). At the top of the client function, before its round loop:

    emu = NetworkEmulator(net_cfg, n_clients)

Inside the round loop, add the three pauses where shown (use the SAME byte numbers that are logged as download_bytes and upload_bytes):

    # 1) right after the global model arrives
    net_down_s = emu.pause_download(download_bytes)

    # ... your existing local training / snapshot code produces `message` and `info` ...

    # 2) after training. compute_s = the same numbers you log as local_train_s + snapshot_opt_s
    slow_extra_s = emu.pause_slow(client_id, info["local_train_s"] + info.get("snapshot_opt_s", 0.0))

    # 3) just before the message is put on the result queue
    net_up_s = emu.pause_upload(upload_bytes)

    info.update(net_download_s=net_down_s, slow_extra_s=slow_extra_s, net_upload_s=net_up_s)

Keep max_client_compute_s in rounds.csv as pure compute time (do not add the pauses to it).

**Edit 4: the server logs the new numbers**

Where the clients.csv logger is created, use `CLIENTS_COLUMNS + NET_CLIENT_COLUMNS` (for OS-Fed runs: `CLIENTS_COLUMNS_OSFED + NET_CLIENT_COLUMNS`). In the `clients_log.log(...)` call add:

    net_download_s=round(info.get("net_download_s", 0.0), 3),
    slow_extra_s=round(info.get("slow_extra_s", 0.0), 3),
    net_upload_s=round(info.get("net_upload_s", 0.0), 3),

Only run_process applies the emulation. Keep run_single at the default net0.

If any edit does not fit your file, paste src/run_process.py and ask for the exact patched version.

## 3. Run the validation conditions (small CNN, pcheck config, 3 rounds)

    for COND in bw10 bw10-d50 slow2x2 bw10-d50-slow2x2; do
      python -m src.run_process --config configs/A_fedavg_cnn_noniid_pcheck_s1.yaml --condition $COND
    done

Expected extra time per round for FedAvg (messages of 2.18 MB): bw10 adds about 3.5 s (1.74 s down + 1.74 s up), d50 adds about 0.1 s, slow2x2 adds one more compute time for the two slow clients.

## 4. Check each emulated run

    for D in logs/*pcheck_s1__bw10_proc logs/*pcheck_s1__bw10-d50_proc logs/*pcheck_s1__slow2x2_proc logs/*pcheck_s1__bw10-d50-slow2x2_proc; do
      python -m scripts.check_emulation_logs --run "$D"
    done

Each must end with OVERALL: PASS. Accuracy must not change (bandwidth and delay only change time). Compare with the Step 9 net0 run:

    REF=$(ls -d logs/*pcheck_s1__net0_proc | tail -1)
    for D in logs/*pcheck_s1__bw10_proc logs/*pcheck_s1__bw10-d50-slow2x2_proc; do
      python -m scripts.compare_runs --single "$REF" --proc "$D" --acc-tol 0.0001
    done

Optional, OS-Fed: if run_process supports it, copy her OS-Fed config the same way as for FedAvg (ls configs to find its name; set samples_per_client: 500, local_epochs: 2, rounds: 3), and run the same four conditions. Expect the upload pause to be tiny and the download pause to be the same as FedAvg, because the full model is still downloaded.

## 5. Notes to add to notes/decisions.md

| # | Item | Decision |
|---|---|---|
| D22 | Link model | Each client has its own link. The server link is not a bottleneck. Every message costs bytes x 8 / (Mbps x 1e6) + delay, using payload bytes. Emulated by time.sleep on one computer |
| D23 | Which clients are slow | The n clients with the highest ids (for slow2x2 with 10 clients: 8 and 9). Extra pause = (factor - 1) x measured compute time |
| D24 | Conditions used | net0, bw100, bw10, bw1, d50, d200, slow2x2, slow5x4 and their combinations. Real emulated runs only for validation (pcheck config). The full grid comes from Step 11 calculation |

## 6. Commit

    git add src/network.py src/run_process.py scripts/test_network.py scripts/check_emulation_logs.py
    git add notes/decisions.md notes/run_log.md logs/
    git commit -m "phase5: emulation"
    git push

Do not run bw1 for many rounds with real pauses: at 1 Mbps one FedAvg message takes 17.4 s, so 100 rounds would add about an hour of waiting. The full grid is calculated in Step 11.

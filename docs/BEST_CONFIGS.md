# Best configs (from the sequential budget sweep, 2026-09-24)

(Tracked copy of `docs/result/visualization/best/index.md`; the pkl/HTML artifacts themselves live there, untracked. Selection rule and sweep tables: `docs/REFACTOR_LOG.md` 2026-09-24 evening entry; the configs are wired into `docs/exp.py` `commands_1`.)

| benchmark | config | volume (baseline) | collars | wall s (baseline) | 3D |
|---|---|---|---|---|---|
| bv_16 | `-s 2 -t 2 -i 1000` | **486.0** (486.0) | 21/21 | 14 (14) | [open](http://localhost:8765/best/bv_16__s2t2i1000_interactive.html) |
| dj_16 | `-s 8 -t 2 -i 1000 --backtrack 1` | **567.0** (729.0) | 31/31 | 24 (17) | [open](http://localhost:8765/best/dj_16__s8t2i1000_bt_interactive.html) |
| ghz_16 | `-s 2 -t 2 -i 1000 --backtrack 1` | **243.0** (891.0) | 1/1 | 14 (61) | [open](http://localhost:8765/best/ghz_16__s2t2i1000_bt_interactive.html) |
| vqe_16 | `-s 4 -t 2 -i 1000 --backtrack 1` | **3645.0** (3888.0) | 82/82 | 214 (152) | [open](http://localhost:8765/best/vqe_16__s4t2i1000_bt_interactive.html) |
| wstate_16 | `-s 8 -t 2 -i 1000 --backtrack 1` | **8019.0** (8262.0) | 74/74 | 184 (159) | [open](http://localhost:8765/best/wstate_16__s8t2i1000_bt_interactive.html) |
| qaoa_16 | `-s 4 -t 2 -i 1000` | **4050.0** (4941.0) | 48/48 | 152 (261) | [open](http://localhost:8765/best/qaoa_16__s4t2i1000_interactive.html) |

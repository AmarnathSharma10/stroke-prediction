# Stroke Labels

This file documents all stroke-type labels used throughout the project — their integer class IDs, the raw annotation strings from each dataset, approximate sample counts, and the natural-language descriptions fed to the BERT-conditioned decoder.

---

## Class Index Summary

| ID | Stroke Name | Short Description |
|----|-------------|-------------------|
| 0 | **Net Shot** | Precise, short-prep shot near the net that just clears the tape |
| 1 | **Defensive / Block** | Reactive stroke under pressure — block, lob, or drive return |
| 2 | **Smash** | Aggressive overhead shot with a steep, fast downward trajectory |
| 3 | **Lob / Lift** | Defensive/neutral lift from the front court to the opponent's back court |
| 4 | **Clear** | High, floating overhead shot aimed deep into the opponent's back court |
| 5 | **Drive** | Flat, fast stroke from mid/front court aimed at the mid/back court |
| 6 | **Drop** | Soft, controlled shot placing the shuttle close to the net |
| 7 | **Push / Rush** | Short-prep shot hit low and deep, or a hard downward strike at the net |
| 8 | **Serve** | Starts the rally; aimed under net height to limit opponent's attack |
| 9 | **Unknown** | Unannotated or ambiguous stroke (often an undocumented serve) |

> **Special value `-1`**: returned by the encoding functions when a raw label string is not found in the mapping table (treated as invalid / ignored during training).

---

## ShuttleSet Dataset Mapping

The primary training dataset. Labels are in Traditional Chinese.

| Class ID | Chinese Label(s) | English Meaning | Approx. Count |
|----------|-----------------|-----------------|---------------|
| 0 | `放小球` | Net shot | ~6 716 |
| 0 | `勾球` | Cross-court net shot | — |
| 1 | `擋小球` | Return net short (defensive) | ~3 836 |
| 1 | `防守回挑` | Reaction lob (defensive) | — |
| 1 | `防守回抽` | Reaction drive (defensive) | — |
| 2 | `殺球` | Smash | ~3 749 |
| 2 | `點扣` | Wrist smash | — |
| 3 | `挑球` | Lob / lift | ~4 614 |
| 4 | `長球` | Clear | ~2 440 |
| 5 | `平球` | Drive | ~1 091 |
| 5 | `小平球` | Front-court drive | — |
| 5 | `後場抽平球` | Back-court drive | — |
| 6 | `切球` | Drop | ~2 929 |
| 6 | `過度切球` | Passive / transition drop | — |
| 7 | `推球` | Push | ~3 021 |
| 7 | `撲球` | Rush / net kill-rush | — |
| 8 | `發短球` | Short serve | ~2 060 |
| 8 | `發長球` | Long serve | — |
| 9 | `未知球種` | Unknown stroke | ~1 095 |

---

## BadmintonDB Dataset Mapping

Secondary dataset (`group_and_encode_badmindb_data_with_player`). Labels are in English.

| Class ID | Raw Label(s) | Notes |
|----------|-------------|-------|
| 0 | `Block`, `Block-Bh` | Forehand and backhand block |
| 1 | `Clear`, `Clear-Bh` | Forehand and backhand clear |
| 2 | `Drive`, `Drive-Bh` | Forehand and backhand drive |
| 3 | `Dropshot`, `Dropshot-Bh` | Forehand and backhand drop |
| 4 | `Net-Kill`, `Net-Kill-Bh` | Forehand and backhand net kill |
| 5 | `Net-Lift`, `Net-Lift-Bh` | Forehand and backhand net lift |
| 6 | `Net-Shot`, `Net-Shot-Bh` | Forehand and backhand net shot |
| 7 | `Smash`, `Smash-Bh` | Forehand and backhand smash |
| 8 | `Serve`, `Flick-Serve` | Standard and flick serve |

> **Note:** `FAULT` labels are explicitly skipped and never encoded.

> **Note:** The BadmintonDB class mapping differs slightly from ShuttleSet (e.g. `Clear` → ID 1 in BadmintonDB vs ID 4 in ShuttleSet). Each dataset uses its own `shot_type_encoding` dictionary inside its respective grouping function.

---

## Natural Language Descriptors (BERT Decoder Input)

The decoder uses BERT token embeddings of stroke descriptions to initialise the target-side input. Two descriptor dictionaries are defined in `Utils/playerID_utils.py`:

### Simple Descriptors (`LMM_simple_descriptor`)

| ID | Description |
|----|-------------|
| 0 | Net shot |
| 1 | Defensive strokes / Block, reaction |
| 2 | Smash, offensiv, point finnish |
| 3 | Lob/lift, high trajectory, from front |
| 4 | Clear, high trajectory, from back |
| 5 | Drive, flat trajectory, midcourt |
| 6 | Drop, short |
| 7 | Push/Rush, short prep, neutral |
| 8 | Serve, Start point |
| 9 | Unknown stroke |

### Detailed Descriptors (`LMM_detailed_descriptor`)

| ID | Description |
|----|-------------|
| 0 | Net shot, a precise shot with short preparation from near the net that enables the shuttlecock to tumble across just slightly above the net. |
| 1 | Defensive strokes / block, a shot used to return or intercept aggressive shots, often in the form of a block drive, lob or net shot. Performed under pressure and thus the stroke has a short preparation. |
| 2 | Smash, hit far over the head; very offensive stroke often with a long preparation attempting to win the point by moving the shuttlecock in a fast downward trajectory. |
| 3 | Lob/lift, hit from under the net in the front court lifting the shuttlecock to the backcourt of the opponent; often a defensive/neutral shot with medium/long preparation. |
| 4 | Clear, a floating stroke hit over the head producing a high soft trajectory aimed towards the backcourt of the opponent's side; often seen as a transport shot. |
| 5 | Drive, produces flat trajectory hit from the front/mid court with short preparation and hit towards middle/backcourt; high-tempo, mostly neutral stroke attempting to gain momentum. |
| 6 | Drop, a floating smooth trajectory attempting to place the shuttle very close to the net. |
| 7 | Push/Rush — the push is a stroke hit semi-softly with short preparation used to send the shuttlecock low and deep into the back of the court. Can also be a hard shot striking the shuttlecock that is too high at the net in a downward trajectory. |
| 8 | Serve, starts the rally and puts the shuttlecock into play; served toward the front service line attempting to keep the shuttle under net height making it difficult for the opponent to attack. |
| 9 | Unknown stroke, a stroke that was not recorded properly by the cameras and could not be annotated. Often occurs at the start of the rally (typically an undocumented serve). |

---

## Label Flow in the Pipeline

```
Raw pickle labels
      │
      ▼
group_and_encode_*_data_with_player()
      │  shot_type_encoding dict → integer class ID (0–9, or -1 if unknown)
      ▼
filter_sequences_by_lengthID()   ← remove rallies shorter than 3 or longer than 30 shots
      ▼
PoseData_Forecast (Dataset)
      │  BertTokenizer tokenises LMM_simple_descriptor[label] for each shot
      ▼
DataLoader  →  RallyTemPose
      │  Decoder input: BERT token IDs + positional encoding + player ID
      ▼
CrossEntropyLoss over 10 classes  +  auxiliary probe loss on encoder output
```


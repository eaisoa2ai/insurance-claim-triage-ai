---
title: Synthetic Claims Dataset — ClaimSight
status: draft
created: 2026-06-16
updated: 2026-06-16
---

# Synthetic Claims Dataset

20 claims tied to real images in `data/images/`. Each claim includes a customer statement, repair estimate, claim history, and expected evaluation outcome.

**Distribution:**
- 12 matching / clean → auto-approve
- 4 slight mismatches → human review
- 2 strong mismatches → human review
- 1 missing / ambiguous evidence → human review
- 1 high-payout / late-reported → human review

---

## Matching Claims — Auto-Approve

---

### CLM-001 ⭐ DEMO SCENARIO 1 — Clean Claim

| Field | Value |
|---|---|
| Image | `data/images/0005.jpg` |
| Damage visible | Minor rear bumper scrape and small dent, silver sedan |
| Estimate | $850 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"Someone must have bumped my car while it was parked in a shopping center lot. I came back and found a small dent and a scrape along the rear bumper. I did not see who did it."

**Expected outcome:**
- Damage assessment: minor rear bumper impact, consistent with parking lot collision
- Consistency score: ~0.92
- Coverage: covered (collision)
- Risk score: ~12 (low)
- Payout: $350 ($850 − $500 deductible)
- Route: **AUTO-APPROVE**

---

### CLM-002 — Door Dent, Parking Garage

| Field | Value |
|---|---|
| Image | `data/images/0003.jpg` |
| Damage visible | Door panel dent, gray sedan, no paint break |
| Estimate | $650 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"Another car door hit mine while I was parked in a multi-storey garage. There is a dent on the driver's door but the paint is not broken."

**Expected outcome:**
- Consistency score: ~0.91
- Coverage: covered (collision)
- Risk score: ~10 (low)
- Payout: $150
- Route: **AUTO-APPROVE**

---

### CLM-003 — Side Door Scrape, Drive-Through

| Field | Value |
|---|---|
| Image | `data/images/0028.jpg` |
| Damage visible | Side door scrape and paint transfer, beige/gold pickup truck |
| Estimate | $480 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | 1 claim (3 years ago, minor fender) |

**Customer statement:**
"I scraped the side of my truck against a concrete pillar while going through a car wash. There is a long paint transfer mark on the door."

**Expected outcome:**
- Consistency score: ~0.88
- Coverage: covered (collision)
- Risk score: ~18 (low)
- Payout: $0 (estimate below deductible — warning issued, no payout)
- Route: **AUTO-APPROVE** (with warning: estimate below deductible)

---

### CLM-004 — Rear-End at Stop Sign

| Field | Value |
|---|---|
| Image | `data/images/0025.jpg` |
| Damage visible | Moderate rear bumper damage, silver sedan, crumpled bumper cover |
| Estimate | $1,800 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"I was stopped at a stop sign when a car rear-ended me. The other driver admitted fault. My rear bumper took the impact."

**Expected outcome:**
- Consistency score: ~0.94
- Coverage: covered (collision)
- Risk score: ~14 (low)
- Payout: $1,300
- Route: **AUTO-APPROVE**

---

### CLM-005 — Rear Collision on Highway

| Field | Value |
|---|---|
| Image | `data/images/0012.jpg` |
| Damage visible | Deep rear bumper dent and crumple, black sedan |
| Estimate | $2,200 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | 1 claim (2 years ago, minor rear-end) |

**Customer statement:**
"I was merging onto the highway when the car behind me failed to brake and hit me from behind. The rear bumper is crumpled and the trunk lid does not close properly."

**Expected outcome:**
- Consistency score: ~0.89
- Coverage: covered (collision)
- Risk score: ~22 (low)
- Payout: $1,700
- Route: **AUTO-APPROVE**

---

### CLM-006 — Fender Dent, Hit-and-Run

| Field | Value |
|---|---|
| Image | `data/images/0016.jpg` |
| Damage visible | Front fender dent, red Jeep Grand Cherokee |
| Estimate | $1,100 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"Someone hit my Jeep in a parking lot and drove off without leaving a note. I noticed the dent on my front fender when I got back to my car."

**Expected outcome:**
- Consistency score: ~0.87
- Coverage: covered (collision / uninsured motorist)
- Risk score: ~16 (low)
- Payout: $600
- Route: **AUTO-APPROVE**

---

### CLM-007 — Side Dent, Parked on Street

| Field | Value |
|---|---|
| Image | `data/images/0017.jpg` |
| Damage visible | Slight side door dent, blue coupe |
| Estimate | $720 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"A delivery van sideswiped my car while I was legally parked on the street. A neighbour witnessed it but the van did not stop."

**Expected outcome:**
- Consistency score: ~0.90
- Coverage: covered (collision / uninsured motorist)
- Risk score: ~11 (low)
- Payout: $220
- Route: **AUTO-APPROVE**

---

### CLM-008 — Front Bumper Scrape, Own Garage

| Field | Value |
|---|---|
| Image | `data/images/0027.jpg` |
| Damage visible | Front bumper scrape and small dent, blue car |
| Estimate | $950 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"I misjudged the distance pulling into my garage and scraped the front bumper against the door frame. Minor damage to the bumper cover."

**Expected outcome:**
- Consistency score: ~0.85
- Coverage: covered (collision — self-caused)
- Risk score: ~14 (low)
- Payout: $450
- Route: **AUTO-APPROVE**

---

### CLM-009 — Rear Panel Scratches, Hit-and-Run

| Field | Value |
|---|---|
| Image | `data/images/0002.jpg` |
| Damage visible | Deep scratches and paint transfer on rear quarter panel, black hatchback |
| Estimate | $1,400 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"I came back to my car in an underground car park and found deep scratches and orange paint transfer on the rear panel. I did not see it happen."

**Expected outcome:**
- Consistency score: ~0.88
- Coverage: covered (collision / hit-and-run)
- Risk score: ~15 (low)
- Payout: $900
- Route: **AUTO-APPROVE**

---

### CLM-010 — Broken Window, Vandalism

| Field | Value |
|---|---|
| Image | `data/images/G0030.jpg` |
| Damage visible | Driver-side window smashed, broken glass visible, red car |
| Estimate | $680 |
| Policy | Comprehensive coverage, $0 glass deductible |
| Prior claims | None |

**Customer statement:**
"My car was broken into overnight. The driver window was smashed and my bag was taken from the back seat. Nothing else inside was damaged."

**Expected outcome:**
- Consistency score: ~0.95
- Coverage: covered (comprehensive — vandalism / theft)
- Risk score: ~10 (low)
- Payout: $680 (no deductible on glass)
- Route: **AUTO-APPROVE**

---

### CLM-011 — Rear-End at Traffic Light

| Field | Value |
|---|---|
| Image | `data/images/S0006.jpg` |
| Damage visible | Significant rear bumper collision dent, silver Infiniti sedan |
| Estimate | $2,800 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | 1 claim (4 years ago) |

**Customer statement:**
"I was at a red light when a car hit me from behind. The impact was strong enough to push me forward. My rear bumper and trunk area are badly dented."

**Expected outcome:**
- Consistency score: ~0.91
- Coverage: covered (collision)
- Risk score: ~20 (low)
- Payout: $2,300
- Route: **AUTO-APPROVE**

---

### CLM-012 — Severe Rear-End, Covered

| Field | Value |
|---|---|
| Image | `data/images/S0017.jpg` |
| Damage visible | Severe rear crumple with taillight destruction, dark sedan |
| Estimate | $4,200 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"I was rear-ended at speed on the motorway. The entire rear of my car is crushed. The other driver's insurer admitted liability but I am claiming through my own policy while it is resolved."

**Expected outcome:**
- Consistency score: ~0.93 (severe damage matches story)
- Coverage: covered (collision)
- Risk score: ~28 (low — clean history, matching story)
- Payout: $3,700
- Route: **AUTO-APPROVE** (under $5,000 limit, clean history)
- *Note: tests that severe-but-matching damage does not alone trigger escalation*

---

## Slight Mismatches — Human Review

---

### CLM-013 — Severity Understated

| Field | Value |
|---|---|
| Image | `data/images/0015.jpg` |
| Damage visible | Heavily crushed driver-side doors and quarter panel, major collision, silver compact |
| Estimate | $3,500 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"A car sideswiped me while I was parked outside my flat. The driver-side door has some damage."

**Expected outcome:**
- Vision findings: severe side impact — both doors crumpled, structural deformation visible
- Consistency score: ~0.48 (image shows far more than "some damage to a door")
- Route: **HUMAN REVIEW**
- Escalation reason: image severity inconsistent with reported minor sideswiping

---

### CLM-014 — Unusual Windshield Damage Pattern

| Field | Value |
|---|---|
| Image | `data/images/D0023.jpg` |
| Damage visible | Multiple large holes in rear windshield, concentrated impact points |
| Estimate | $1,200 |
| Policy | Comprehensive coverage, $250 glass deductible |
| Prior claims | 1 claim (1 year ago, windshield) |

**Customer statement:**
"A hailstorm damaged my rear windshield. There are multiple impact points across the glass."

**Expected outcome:**
- Vision findings: multiple large-diameter holes — pattern more consistent with targeted impacts than hail distribution (hail produces small, uniform pitting)
- Consistency score: ~0.52
- Prior windshield claim within 12 months: mild risk signal
- Route: **HUMAN REVIEW**
- Escalation reason: damage pattern inconsistent with hail; prior similar claim

---

### CLM-015 — Door Ding or Collision?

| Field | Value |
|---|---|
| Image | `data/images/0004.jpg` |
| Damage visible | Both driver-side doors severely crumpled, structural deformation, blue car |
| Estimate | $4,800 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"Another car door hit mine while I was parked in a supermarket car park. The door is dented."

**Expected outcome:**
- Vision findings: major T-bone-style side collision — both doors destroyed, not a door ding
- Consistency score: ~0.21 (extreme severity mismatch)
- Payout would be $4,300 — near but under limit; escalation driven by mismatch, not payout
- Route: **HUMAN REVIEW**
- Escalation reason: image shows major collision incompatible with reported door ding

---

### CLM-016 — Storm Debris or Rollover?

| Field | Value |
|---|---|
| Image | `data/images/G0021.jpg` |
| Damage visible | Roof heavily crushed, all windows destroyed, white car at accident scene |
| Estimate | $6,000 |
| Policy | Comprehensive coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"During a bad storm, a large tree branch fell on my car and caused significant damage to the roof and windows."

**Expected outcome:**
- Vision findings: roof crushed downward, consistent with rollover or major overhead impact; scene context (highway, other cars) suggests accident, not parked storm damage
- Consistency score: ~0.51
- Payout $5,500 — above auto-approve limit
- Route: **HUMAN REVIEW**
- Escalation reason: payout above limit + image context inconsistent with parked storm damage

---

## Strong Mismatches — Human Review

---

### CLM-017 ⭐ DEMO SCENARIO 2 — Image-Story Mismatch

| Field | Value |
|---|---|
| Image | `data/images/F0022.jpg` |
| Damage visible | **None** — black Jeep Grand Cherokee shows no visible damage |
| Estimate | $4,500 |
| Policy | Collision coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"I was involved in a front-end collision at an intersection. The impact caused significant structural damage to the front bumper and engine bay. The car is driveable but the front end is damaged."

**Expected outcome:**
- Vision findings: no visible damage detected anywhere on the vehicle
- Consistency score: ~0.05
- `image_claim_mismatch = True`
- Route: **HUMAN REVIEW**
- Escalation reason: vision model finds zero damage; claim describes significant structural damage

---

### CLM-018 — Fire Damage Claimed as Collision

| Field | Value |
|---|---|
| Image | `data/images/S0022.jpg` |
| Damage visible | Extensive fire / burn damage on rear of white vehicle |
| Estimate | $8,500 |
| Policy | Collision coverage only (no comprehensive / fire coverage) |
| Prior claims | 2 claims (18 months ago, 6 months ago) |

**Customer statement:**
"I was rear-ended at an intersection. The collision caused significant damage to the rear of my vehicle."

**Expected outcome:**
- Vision findings: fire and burn damage detected — no collision deformation
- Consistency score: ~0.08 (wrong damage type entirely)
- Coverage: excluded — fire damage not covered under collision-only policy
- `image_claim_mismatch = True`, `coverage_decision = "excluded"`
- Prior claims pattern: 3 claims in 24 months (including this one) — elevated risk
- Route: **HUMAN REVIEW**
- Escalation reason: image shows fire damage; policy excludes fire; claimed as rear-end collision

---

## Missing / Ambiguous Evidence — Human Review

---

### CLM-019 — Ambiguous Evidence

| Field | Value |
|---|---|
| Image | `data/images/F0012.jpg` |
| Damage visible | Red car with bricks/masonry fallen on and around it, front end heavily loaded with debris |
| Estimate | $5,200 |
| Policy | Comprehensive coverage, $500 deductible |
| Prior claims | None |

**Customer statement:**
"My car was damaged in a car park. I am not sure exactly what happened — I came back and found it like this."

**Expected outcome:**
- Vision findings: extensive masonry / structural debris — consistent with wall collapse, not a standard parking lot collision
- Customer statement is vague: "I am not sure what happened" — IntakeAgent flags completeness
- `IntakeResult.confidence ≈ 0.55` (incomplete statement)
- Coverage ambiguous: wall collapse may fall under comprehensive / act of God or liability — PolicyAgent flags partial
- Route: **HUMAN REVIEW**
- Escalation reason: vague customer statement + unusual damage type + confidence below floor

---

## High Payout / Late Reported — Human Review

---

### CLM-020 ⭐ DEMO SCENARIO 3 — High-Risk, High-Payout

| Field | Value |
|---|---|
| Image | `data/images/S0016.jpg` |
| Damage visible | Blue Smart car, front end completely destroyed — total loss |
| Estimate | $14,500 |
| Policy | Collision coverage, $500 deductible, $15,000 cap |
| Prior claims | 3 claims in the last 18 months (minor collision, rear-end, windshield) |
| Days since incident | 45 days (late reporting) |

**Customer statement:**
"My car was completely wrecked in a head-on collision. I have been dealing with injuries and only now have the chance to file the claim."

**Expected outcome:**
- Vision findings: total-loss — front crumple zone destroyed, engine bay exposed
- Consistency score: ~0.82 (image matches story for severity)
- Risk score: ~88 (critical)
  - Prior claims: 3 in 18 months (+30 pts)
  - Late reporting: 45 days (+20 pts)
  - High payout (+15 pts)
  - Consistency partially mitigates (-17 pts)
- `risk_category = "critical"`
- Payout: $14,000 (estimate $14,500 − $500 deductible) — far above auto-approve limit
- Route: **HUMAN REVIEW**
- Escalation reasons: risk score critical + payout above limit + late reporting + high prior-claim frequency

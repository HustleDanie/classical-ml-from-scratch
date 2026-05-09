# Expert Scenarios Library

A library of realistic business problems for classical ML, organized so you can go from "I just got a brief" to "here's the closest playbook" in under 5 minutes.

---

## How This Library Is Organized

```
expert_scenarios/
├── classification/
│   ├── catalog/                    -- ~80 classification scenarios (deep dives + planned entries)
│   │   └── CATALOG.md              -- the index: every scenario summarized
│   └── from_brief_to_solution/
│       └── METHODOLOGY.md          -- how to read a brief and find the matching scenario
│
└── regression/
    ├── catalog/                    -- ~80 regression scenarios
    │   └── CATALOG.md
    └── from_brief_to_solution/
        └── METHODOLOGY.md
```

---

## How to Use This Library

You'll come here in one of three modes:

### Mode 1: I have a brief and a dataset, what do I do?

1. Open the methodology guide for your problem type:
   - [Classification methodology](classification/from_brief_to_solution/METHODOLOGY.md)
   - [Regression methodology](regression/from_brief_to_solution/METHODOLOGY.md)
2. Walk the 6-step "brief reading" protocol.
3. Land on a catalog category, then drill to the closest 1–3 scenarios.
4. Read the deep dive for the closest scenario; adapt to your specifics.

### Mode 2: I want to study a specific scenario

Open the catalog for your problem type and skim:

- [Classification catalog](classification/catalog/CATALOG.md)
- [Regression catalog](regression/catalog/CATALOG.md)

Each entry summarizes the archetype, recommended approach, primary metric, and links to a full walkthrough if one exists.

### Mode 3: I want to stress-test myself

Pick any scenario. Read **only the brief**. Close the file. Answer on paper:

1. What's the problem type?
2. Which algorithms would I try, and why?
3. What's my pipeline?
4. What metric matters, and why?

Then read the rest to grade yourself.

---

## What's a "Deep Dive" vs a "Planned" Entry?

Each catalog entry has a **status tag**:

- **Deep dive available** — the full ~500-line walkthrough exists in this folder. Click the link.
- **[planned]** — the archetype is listed in the catalog but the full walkthrough is on the roadmap. The catalog entry itself still gives you enough (archetype, brief signal, data shape, recommended approach, primary metric, watch-outs) to start working.

There are 9 deep dives today and ~150 planned entries. Deep dives are added in batches over time.

---

## Out of Scope (For Now)

- **Clustering scenarios** — see [EXPERT_SCENARIO_9_ECOMMERCE_SEGMENTATION.md](../EXPERT_SCENARIO_9_ECOMMERCE_SEGMENTATION.md) at the repo root for the one existing clustering example.
- **Dimensionality reduction scenarios** — covered indirectly by the existing [HOW_IT_WORKS_11_PCA.md](../HOW_IT_WORKS_11_PCA.md) and the regression / classification scenarios that apply PCA inside their pipelines.

---

## See Also

- [START_HERE.md](../START_HERE.md) — the 4-week learning path that lands here in week 4
- [PROBLEM_TYPES_GUIDE.md](../PROBLEM_TYPES_GUIDE.md) — the underlying 4 problem types
- [MODEL_SELECTION_GUIDE_CLASSIFICATION.md](../MODEL_SELECTION_GUIDE_CLASSIFICATION.md) — the 8 framing questions for classification
- [MODEL_SELECTION_GUIDE_REGRESSION.md](../MODEL_SELECTION_GUIDE_REGRESSION.md) — the 6 framing questions for regression
- [ML_PIPELINE_GUIDE.md](../ML_PIPELINE_GUIDE.md) — the universal 12-phase pipeline every scenario follows

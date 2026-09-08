# Dynamic Reluctance for the 5G Healthcare Double Auction

---

## In one paragraph

The existing draft is a complete **market**: hospitals (HSPs) buy uplink from base stations (BSs), a third party runs a double auction, and nobody has to reveal their true costs. Two private numbers sit inside that market: **preference** (how much a hospital wants a given cell, because of critical patients there) and **reluctance** (how painful it is for that cell to sell extra capacity). We have already made **preference** smart, using LSTM forecasts of patient vitals. **Reluctance is still treated as a fixed cost.** This proposal is to make reluctance move with the real radio network—load, signal quality, backhaul, impact on ordinary mobile users—so the *seller* side of the auction is as adaptive as the *buyer* side. The auction rules, pricing, and economic properties stay exactly as in the draft.

---



## What is already done

The draft assumes the hospital **already knows** who is critical and where they sit. We replaced that assumption with a learning pipeline:

- Forecast vitals a few minutes ahead (LSTM).
- Turn those forecasts into a 0–1 criticality score per patient.
- Add up scores under each (hospital, base station) pair → **dynamic preference**.
- Run the same kind of auction; more critical demand gets more rate.

So the **buyer** now reacts to predicted health, not only to a snapshot count.

---



## What is still missing

The draft also says the **base station** has a private **reluctance**: serving healthcare traffic can degrade its own subscribers and costs more when the channel is bad.

In our current prototype that pain is a **constant**. Every cell, every hospital, every minute: same selling cost. The network does not become “less willing to sell” when it is congested or when those patients have a weak link.

If we only smarten preference:

- The hospital shouts louder for a cell that will soon have many critical patients.
- The auction **pushes more traffic onto that cell**.
- If that cell is already full, ordinary users suffer, and the draft’s own motivation (protect native traffic) is ignored.

**Preference without reluctance is a one-sided smart system.**

---



## Why this extension is necessary

A smart 5G healthcare network must answer two questions every few minutes:


| Side         | Question                                                                                   | Our status                  |
| ------------ | ------------------------------------------------------------------------------------------ | --------------------------- |
| Hospital     | “Where will my sickest patients be, and how badly will they need uplink?”                  | Addressed (LSTM preference) |
| Base station | “Can I absorb that extra traffic without harming my own users or wasting radio resources?” | **Not yet**                 |


Reluctance is the second answer. It should rise when:

- the cell is busy,
- the wireless link to those healthcare users is poor,
- the backhaul is packed,
- or ordinary mobile users’ quality is already dropping.

Then the base station’s **bid** in the auction naturally says “this slice is expensive,” and the market **gives less rate** on that link—or the hospital buys from another cell instead. Same auction; better information.

---



## What we are *not* changing

- Double auction, third-party auctioneer, bid-and-allocate loop  
- Incentive compatibility, individual rationality, budget balance, efficiency  
(these still hold **inside each time slot**, with preference and reluctance held fixed during that short loop, exactly as a single shot in the draft)

We only change **how the base station’s private cost is produced** before each slot.

---



## How we will model it (idea only)

Reluctance is **not** in the Kaggle patient file. That file has heart rate and oxygen, not cell load. So we **define** a sensible “true” reluctance from radio measurements we can simulate (how full the cell is, how strong the signal is, backhaul, interference, quality of ordinary users). Then we **train a model to predict that value** from what a base station could actually observe.

Plain-language pipeline:

1. Build a **training table**: many time slots × each (cell, hospital) pair, with radio features and the defined reluctance target.
2. Train a predictor (simple model first, stronger models later).
3. Each slot: update hospital preference (LSTM, already done) **and** cell reluctance (new).
4. Freeze both, run the **same** auction, record rates, payments, and bids.

Patient data trains preference. Radio traces train reluctance. They are not mixed into one messy model.

---



We will go **step by step**, not jump to the heaviest model:

1. **Formula / small model** — reluctance as a clear function of load and signal quality (easy to explain, baseline).
2. **Standard ML (e.g. gradient boosting)** — same features, better fit, still interpretable enough.
3. **Sequence model (LSTM-style)** — if congestion is a *trend* over minutes, same reason we used LSTM for vitals.
4. **Graph model (optional)** — if neighbouring cells interfere, one cell’s pain depends on the next; a graph network captures that.

The “best” first paper increment is (1)+(2) plus auction plots. (3) and (4) are the natural follow-on if we want a full smart-RAN story.

We will **not** use reinforcement learning to invent reluctance in v1. In the draft, reluctance is a **physical cost**, not a strategy to game the market. Bidding stays truthful; we only estimate the cost more honestly.

---



## Expected results (same three figures as the draft)

The mentor already asked for:

1. **Convergence** — social welfare still reaches the maximum.
2. **Demand vs response gap** — hospital ask and cell offer still meet (gap → 0).
3. **Evolution of bids** — hospital bids and **base-station bids** settle.

The **new** comparison: **fixed reluctance vs dynamic reluctance**.

What should appear: when a cell is loaded or the link is weak, **that cell’s bids go up** and it **sells less**; critical demand **shifts** toward a healthier cell. Convergence and zero gap should **still** hold. That is the evidence that the extension is compatible with the original mechanism.

---






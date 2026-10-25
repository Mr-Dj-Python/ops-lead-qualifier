# Lead Scoring Guide (for the Business Team)

**Purpose:** explain, in plain language, how the system scores each lead, why each factor matters, and how scores become decisions. No technical knowledge is needed.

**Product context:** we sell operations software (inventory, warehouse, fleet and supply planning) to companies that move, make or sell physical goods, with 50-500 employees.

**Status:** these weights are our best starting assumptions. Please review and challenge them (see section 7).

---

## 1. How a lead is scored (in one minute)

Every lead gets a score from **1 to 10** on four factors. Each factor counts for a share of the final score:

| Factor | Share of final score | Question it answers |
|---|---|---|
| **Industry fit** | **40%** | Does this company's business need what we sell? |
| **Company size** | **25%** | Is the company the right size for our product? |
| **Lead source** | **20%** | How strong is the buying intent behind how they reached us? |
| **Recent engagement** | **15%** | How recently did they interact with us? |

**Final score = (Industry x 0.40) + (Size x 0.25) + (Source x 0.20) + (Engagement x 0.15)**

---

## 2. Why these weights

- **Industry fit (40%)** is the biggest because it separates real buyers from people who will never need the product. A perfect lead in the wrong industry is still a waste of a sales call.
- **Company size (25%)** comes next. Very small companies cannot afford us or do not need the scale. Very large ones need enterprise deals and longer sales cycles.
- **Lead source (20%)** reflects intent. Someone who asked for a demo is much more likely to buy than someone who received a cold LinkedIn message.
- **Recent engagement (15%)** is the smallest because it can be fixed with a good follow-up. A cold lead in the right industry is still worth contacting.

---

## 3. Scoring tables

### 3.1 Industry fit (40%)
Judged by the AI against the table below, then checked by the team.

| Fit level | Score | Industries |
|---|---|---|
| **Best fit** | 9-10 | Logistics, Supply Chain, Manufacturing, Retail, Automotive, Trade |
| **Hard fit** | 4-6 | Construction, Energy, Agriculture, Healthcare, Environmental, Consumer goods |
| **No fit** | 1-3 | Software and SaaS, Cybersecurity, AI/ML, Analytics, Finance and FinTech, Insurance, Media and Entertainment, Telecom, Legal, Consulting, Marketing, Education, Government, Non-profit, Hospitality, Events, Sports, Travel, Real Estate, general Services, and similar |

**Special case: look-alike vendors.** Some companies are labeled "Software" but sell logistics or supply chain tools (for example "Logistics Optimization" or "Supply Chain Analytics"). They are competitors or partners, not buyers, so they score low.
**Special case: hidden buyers.** A company with a different label but real physical operations (for example "Pharmaceutical Distribution", labeled Healthcare) can score higher than its label suggests.

### 3.2 Company size (25%)
| Employees | Score | Why |
|---|---|---|
| 50 - 500 | 10 | Our sweet spot |
| 20 - 49 or 501 - 1,000 | 7 | Workable, a little off target |
| 10 - 19 or 1,001 - 5,000 | 4 | Too small to need us, or too big for a standard deal |
| Under 10 or over 5,000 | 1 | Not a realistic customer |
| Missing | 5 | Neutral; we do not penalize missing data |

### 3.3 Lead source (20%)
| Source | Score | Why |
|---|---|---|
| Inbound demo request | 10 | They asked to see the product |
| Referral | 9 | Pre-trusted introduction |
| Sales call | 8 | Direct conversation already happened |
| Webinar attendee | 6 | Showed interest in the topic |
| Content download | 5 | Early research stage |
| LinkedIn outreach | 3 | We contacted them; they did not ask |
| Unknown | 4 | Cannot judge intent |

### 3.4 Recent engagement (15%)
Measured from the report date to the lead's last interaction.

| Days since last interaction | Score |
|---|---|
| 14 or fewer | 10 |
| 15 - 30 | 7 |
| 31 - 90 | 4 |
| Over 90 or missing | 1 |

---

## 4. From score to decision

| Final score | Decision | What the sales team does |
|---|---|---|
| **7.0 and above** | **Qualified** | Contact in priority order (highest score first). A personalized message is drafted. |
| **5.0 - 6.9** | **Review** | A person decides. These are "too close to call"; the system does not guess. |
| **Below 5.0** | **Rejected** | No outreach. The reason is recorded (the lowest-scoring factor). |

**Safety rule:** a lead in a "No fit" industry can never be Qualified, however strong its other scores. If it is also under 10 or over 5,000 employees, it is Rejected outright.

**Priority order** among Qualified leads: highest score first; if tied, most recently active first.

---

## 5. Worked examples

Report date: 21 Jan 2024.

**Example 1: Warehouse Automation** (300 employees, Logistics, Referral, last contact 6 Jan = 15 days)
| Factor | Score | Weight | Points |
|---|---|---|---|
| Industry | 10 | 40% | 4.00 |
| Size | 10 | 25% | 2.50 |
| Source | 9 | 20% | 1.80 |
| Engagement | 7 | 15% | 1.05 |
| **Final** | | | **9.35 -> Qualified** |

**Example 2: Construction Management** (320 employees, Construction, LinkedIn outreach, last contact 29 Dec = 23 days)
| Factor | Score | Weight | Points |
|---|---|---|---|
| Industry (hard fit) | 5 | 40% | 2.00 |
| Size | 10 | 25% | 2.50 |
| Source | 3 | 20% | 0.60 |
| Engagement | 7 | 15% | 1.05 |
| **Final** | | | **6.15 -> Review** |

**Example 3: Government Agency** (800 employees, Government, Content download, last contact 10 Dec = 42 days)
| Factor | Score | Weight | Points |
|---|---|---|---|
| Industry (no fit) | 2 | 40% | 0.80 |
| Size | 7 | 25% | 1.75 |
| Source | 5 | 20% | 1.00 |
| Engagement | 4 | 15% | 0.60 |
| **Final** | | | **4.15 -> Rejected** |

---

## 6. What the system does not know
- It has **no budget, timeline or decision-maker data**, so those are not scored.
- Weights are **assumptions**, not learned from past deals. Once we know which leads actually closed, the weights should be adjusted.
- Industry fit is an AI judgment on free-text labels, so edge cases can differ slightly between runs. This is why borderline leads go to Review.
- Missing data is scored neutrally, never as a penalty.

---

## 7. Questions for the business team to confirm
1. Is **Industry fit at 40%** right, or should sales intent (source) matter more?
2. Is **50-500 employees** really our best size range?
3. Are the six **best-fit industries** correct? Is anything missing (for example Wholesale or Food and Beverage)?
4. Should **LinkedIn outreach** leads (score 3) ever be Qualified, or only Review?
5. Is the **7.0 / 5.0** cutoff right for how many leads the team can contact (current capacity: about 60 a month)?
6. Is a **14-day** window the right definition of "fresh" engagement?

*Any change agreed here is a one-line edit in the system's settings file; no code changes are needed.*

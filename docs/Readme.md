# 📋 Business Documentation — Pump Failure Prediction MVP

## link: https://pump-failure-prediction-bhfv5dlamfpdvpobxuqofk.streamlit.app/

## 1. Problem Definition

### Business Context
Hydraulic piston pumps are critical components in copper mining production
processes. Unplanned pump failures cause:

- **Production stoppage** — direct revenue loss
- **Emergency maintenance costs** — 3-5x more expensive than planned maintenance
- **Safety risks** — hydraulic fluid leaks under high pressure
- **Cascading failures** — downstream equipment damage

### Problem Statement
Currently, pump maintenance is performed on a **fixed schedule** (time-based),
regardless of the actual condition of the equipment. This approach leads to:

| Issue | Impact |
|---|---|
| Premature maintenance | Unnecessary cost and downtime |
| Late maintenance | Unplanned failures and production loss |
| No early warning | Reactive instead of proactive response |

### Proposed Solution
A **machine learning classification model** that analyzes real-time sensor
readings and diagnoses the operational state of the pump, enabling:

- Early detection of valve plate degradation
- Transition from time-based to **condition-based maintenance**
- Prioritization of maintenance actions based on failure severity

---

## 2. Business Objective

> **Primary objective:** Detect pump valve plate failures before they cause
> unplanned production stoppages, reducing emergency maintenance costs and
> increasing equipment availability.

### Scope
- **Equipment:** Hydraulic piston pumps used in copper ore processing
- **Component:** Valve plate (most frequent failure mode)
- **Deployment:** MVP — batch prediction via CSV upload on Streamlit dashboard
- **Future scope:** Real-time inference integrated with SCADA/OT systems

---

## 3. Success Metrics

### 3.1 Technical Metrics

| Metric | Target | Achieved | Status |
|---|---|---|---|
| F1-Macro (test set) | > 90% | **99.72%** | ✅ |
| ROC-AUC (test set) | > 95% | **100.0%** | ✅ |
| F1 — Simulated Failure 1 | > 85% | **99.61%** | ✅ |
| F1 — Simulated Failure 2 | > 85% | **99.58%** | ✅ |
| False Negative Rate (Failure→Normal) | < 5% | **~0.0%** | ✅ |

> **Why F1-Macro as primary metric?**
> Accuracy would be misleading with imbalanced classes. F1-Macro penalizes
> equally errors across all classes, including rare failure states.

> **Why False Negative Rate is critical?**
> A failure classified as Normal (missed detection) is the most dangerous
> error — it means the maintenance team receives no alert when action is needed.

### 3.2 Business Metrics

| Metric | Description | Target |
|---|---|---|
| **Mean Time Between Failures (MTBF)** | Average operating time between failures | Increase by 20% |
| **Planned vs Unplanned Maintenance Ratio** | % of maintenance actions that were planned | > 80% planned |
| **Early Detection Rate** | % of failures detected before functional failure | > 90% |
| **False Alarm Rate** | % of Normal classified as Failure | < 5% |

### 3.3 Definition of Done
The MVP is considered successful when:
- [x] Model F1-Macro > 90% on held-out test set
- [x] Dashboard operational with CSV upload and visual alerts
- [x] Zero False Negatives (Failure → Normal) on test set
- [x] All artifacts versioned and reproducible via MLflow
- [ ] Validated on real production pump data (next phase)

---

## 4. Stakeholders

| Role | Interest | Involvement |
|---|---|---|
| Maintenance Engineer | Receives alerts, plans interventions | Primary user |
| Maintenance Manager | KPIs, cost reduction | Dashboard consumer |
| OT/IT Team | System integration, data pipeline | Technical integration |
| Data Science Team | Model development and monitoring | Owner |

---

## 5. Risks and Limitations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset from lab, not production | High | High | Validate with real pump data |
| Sensor drift over time | Medium | High | Implement model monitoring |
| New failure modes not in training | Medium | High | Periodic retraining |
| Single pump type generalization | Medium | Medium | Test on different pump models |


## 6. Scope Limitation — Equipment Applicability

### Current MVP Scope
The model was trained on data from a **hydraulic piston pump** (axial piston,
45 cm³/rev, 37 kW, 1485 RPM). It is applicable to:
- Hydraulic power units
- Hydraulic drive systems in heavy mining equipment
- Any axial piston pump with valve plate as critical component

### Out of Scope (requires new dataset)
| Equipment | Type | Main Failure Mode | Status |
|---|---|---|---|
| Higra Anfíbia | Centrifugal | Impeller wear, cavitation | ❌ Not covered |
| Weir Warman | Centrifugal slurry | Liner/impeller wear | ❌ Not covered |
| KSB Meganorm/MegaCPK | Centrifugal | Seal failure, cavitation | ❌ Not covered |

### Next Steps
1. Survey hydraulic piston pumps in current copper mining operations
2. Identify centrifugal pump failure data availability
3. Plan Phase 2 dataset collection for centrifugal pumps

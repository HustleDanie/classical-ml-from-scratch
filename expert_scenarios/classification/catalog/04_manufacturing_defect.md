# Expert Scenario 4: Manufacturing Defect Detection from Sensor Data

> **Complexity:** Multiclass classification (6 defect types + OK), high-dimensional time-series sensor data, extreme class imbalance (defects are rare), real-time inference, false negative = defective product ships to customer, cost-sensitive decision making.

---

## The Brief

A semiconductor factory has 48 sensors monitoring each chip during a 3-minute fabrication process (temperature, pressure, vibration, chemical flow rates, etc.). Each sensor records 180 readings (1 per second). They produce 50,000 chips/day. Currently, 2.1% have defects caught by end-of-line optical inspection (slow, expensive). The factory wants a model that detects defects from sensor data DURING fabrication, so defective chips can be pulled before expensive final processing. There are 6 defect types (contamination, alignment error, temperature excursion, chemical imbalance, vibration fault, electrical fault) plus "OK". Each defect type requires a different corrective action, so they need multiclass -- not just "defect or not."

This is complex because: 48 sensors x 180 time steps = 8,640 raw features per chip, defects are rare (2.1%), 6 defect types have different frequencies (0.8% down to 0.05%), time-series patterns within each sensor matter (spike at second 45 vs gradual drift), and inference must be fast enough for production line speed.

---

## Step 1: Define the Problem Type

```
Type:           Multiclass Classification (7 classes: OK + 6 defect types)
Primary Metric: Macro F1 (balanced across all defect types)
Secondary:      Per-class recall (must catch rare defects)
Business Goal:  Catch 95%+ of defects before final processing
Constraint:     Inference < 50ms per chip (production line speed)
Cost Matrix:
  - Miss a defect (FN): $120 (wasted final processing + warranty claim)
  - False alarm (FP):   $8 (chip re-inspected manually, 30 seconds)
  - Catching a defect:  $0 (saved money)
```

**Class distribution:**
```
OK:                  97.9%  (48,950 chips/day)
Contamination:       0.8%   (400/day)
Alignment Error:     0.5%   (250/day)
Temp Excursion:      0.3%   (150/day)
Chemical Imbalance:  0.2%   (100/day)
Vibration Fault:     0.15%  (75/day)
Electrical Fault:    0.05%  (25/day) -- rarest, hardest to detect
```

---

## Step 2: Understand the Data

```
Training data: 6 months of production = 9,100,000 chips
Each chip: 48 sensors x 180 seconds = 8,640 raw readings
Label: defect type (from post-production optical inspection)
Total raw matrix: 9.1M rows x 8,640 columns = 78.6 BILLION data points

Sensor groups:
- Temperature sensors (12): chamber temp, wafer surface temp, coolant temp, etc.
- Pressure sensors (8): chamber pressure, gas line pressures
- Chemical flow sensors (10): flow rates of 10 different chemicals
- Vibration sensors (6): XYZ acceleration on 2 mounting points
- Electrical sensors (8): voltage, current, impedance at different stages
- Environment sensors (4): humidity, ambient temp, particle count, air flow
```

**Expert thinking:** 8,640 features per chip is WAY too many. Most ML models will overfit or take forever. We need to extract meaningful statistical features from the time series rather than using raw readings. Domain knowledge tells us that defects show up as:
- **Contamination**: particle count spikes
- **Alignment error**: vibration pattern change in first 30 seconds
- **Temperature excursion**: temperature deviates from setpoint mid-process
- **Chemical imbalance**: flow rate ratios drift over time
- **Vibration fault**: vibration amplitude increases, specific frequency appears
- **Electrical fault**: impedance measurement shows discontinuity

---

## Step 3-4: EDA & Cleaning

```python
# Raw data check
# Found 0.02% of sensor readings are NaN (sensor dropout)
# Forward-fill within each chip's time series (sensor briefly offline)
# 14 chips have >50% NaN (major sensor failure) -- remove from training

# Sensor calibration drift: sensor 7 (chamber pressure) shows gradual offset
# over month 4-5. Recalibrated at month 5 boundary.
# Solution: normalize each sensor relative to its mean over the past 24 hours
# This removes calibration drift without losing the within-chip patterns

# Label quality check:
# Optical inspection has 3% false negative rate (misses some defects)
# This means ~3% of our "OK" labels are actually defective -- noisy labels
# We can't fix this, but we should:
# 1. Not expect better than 97% recall (ceiling due to label noise)
# 2. Consider label smoothing during training
```

---

## Step 5: Feature Engineering (THE Critical Step)

```python
# ================================================================
# STRATEGY: Extract statistical features from each sensor's time series
# Instead of 8,640 raw readings, create ~500 meaningful features
# ================================================================

def extract_sensor_features(sensor_readings, sensor_name):
    """Extract features from one sensor's 180-second time series."""
    features = {}

    # === Basic Statistics ===
    features[f'{sensor_name}_mean'] = np.mean(sensor_readings)
    features[f'{sensor_name}_std'] = np.std(sensor_readings)
    features[f'{sensor_name}_min'] = np.min(sensor_readings)
    features[f'{sensor_name}_max'] = np.max(sensor_readings)
    features[f'{sensor_name}_range'] = np.max(sensor_readings) - np.min(sensor_readings)
    features[f'{sensor_name}_median'] = np.median(sensor_readings)
    features[f'{sensor_name}_skew'] = scipy.stats.skew(sensor_readings)
    features[f'{sensor_name}_kurtosis'] = scipy.stats.kurtosis(sensor_readings)

    # === Percentiles (capture distribution shape) ===
    for pct in [5, 25, 75, 95]:
        features[f'{sensor_name}_p{pct}'] = np.percentile(sensor_readings, pct)

    # === Trend Features (is the sensor drifting?) ===
    x = np.arange(len(sensor_readings))
    slope, intercept = np.polyfit(x, sensor_readings, 1)
    features[f'{sensor_name}_slope'] = slope
    # Positive slope = increasing over process. Temp should be stable -- positive slope = excursion

    # === Phase Features (different behavior at start/middle/end of process) ===
    third = len(sensor_readings) // 3
    features[f'{sensor_name}_phase1_mean'] = np.mean(sensor_readings[:third])
    features[f'{sensor_name}_phase2_mean'] = np.mean(sensor_readings[third:2*third])
    features[f'{sensor_name}_phase3_mean'] = np.mean(sensor_readings[2*third:])
    features[f'{sensor_name}_phase_diff_12'] = features[f'{sensor_name}_phase2_mean'] - features[f'{sensor_name}_phase1_mean']
    features[f'{sensor_name}_phase_diff_23'] = features[f'{sensor_name}_phase3_mean'] - features[f'{sensor_name}_phase2_mean']

    # === Anomaly Features ===
    # Number of readings beyond 2 std from mean
    threshold = np.mean(sensor_readings) + 2 * np.std(sensor_readings)
    features[f'{sensor_name}_num_spikes'] = np.sum(np.abs(sensor_readings) > threshold)
    # Max consecutive readings above threshold (sustained anomaly vs transient spike)
    above = (np.abs(sensor_readings) > threshold).astype(int)
    features[f'{sensor_name}_max_consecutive_spikes'] = max(
        (sum(1 for _ in g) for k, g in itertools.groupby(above) if k), default=0
    )

    # === Rate of Change Features ===
    diff = np.diff(sensor_readings)
    features[f'{sensor_name}_max_rate_of_change'] = np.max(np.abs(diff))
    features[f'{sensor_name}_mean_rate_of_change'] = np.mean(np.abs(diff))
    # Sudden jumps in sensor readings indicate faults

    # === Frequency Domain (for vibration sensors) ===
    if 'vibration' in sensor_name:
        fft_vals = np.abs(np.fft.rfft(sensor_readings))
        features[f'{sensor_name}_dominant_freq'] = np.argmax(fft_vals[1:]) + 1
        features[f'{sensor_name}_fft_energy'] = np.sum(fft_vals**2)
        features[f'{sensor_name}_high_freq_ratio'] = np.sum(fft_vals[30:]) / (np.sum(fft_vals) + 1e-10)
        # High-frequency vibration = potential mechanical fault

    return features

# Apply to all 48 sensors for each chip
# Result: ~500 features per chip (instead of 8,640)

# ================================================================
# CROSS-SENSOR FEATURES (relationships between sensors)
# ================================================================

# Temperature gradient (should be stable -- large gradient = problem)
features['temp_gradient_max'] = max_temp_sensor - min_temp_sensor

# Chemical ratio stability (flow_rate_A / flow_rate_B should be constant)
features['chem_ratio_AB_std'] = np.std(flow_A / (flow_B + 0.001))
# High std = ratio is unstable = chemical imbalance defect

# Pressure-temperature correlation (should be high due to gas laws)
features['pressure_temp_corr'] = np.corrcoef(pressure_readings, temp_readings)[0, 1]
# Correlation < 0.8 = something unusual happening

# Vibration vs machine stage synchronization
features['vibration_sync_score'] = cross_correlation(vibration, expected_pattern)
# Low sync = alignment error

# Electrical impedance at critical moments (seconds 30, 90, 150)
for t in [30, 90, 150]:
    features[f'impedance_at_{t}s'] = impedance_readings[t]
```

**Expert insight:** We reduced 8,640 features to ~500 by extracting statistical summaries. But these aren't random statistics -- each is motivated by physics:
- **Slope** catches gradual drifts (temperature excursion)
- **Spike count** catches contamination events
- **FFT features** catch mechanical resonance (vibration faults)
- **Phase differences** catch process stage anomalies
- **Cross-sensor correlations** catch system-level failures

This is where domain knowledge multiplies your ML performance by 2-3x.

---

## Step 6-8: Feature Selection, Preprocessing, Split

```python
# Feature selection with MI:
# Top features: particle_count_num_spikes (contamination), vibration_X_fft_energy
# (vibration fault), temp_chamber_slope (temp excursion), chem_ratio_AB_std (chemical)
# Each defect type has its own "signature features"

# Removed 80 features with MI < 0.005 -> Final: 420 features

# Preprocessing: StandardScaler for all (LightGBM doesn't need it, but SVM/LR do)

# Time-based split: Train months 1-5, Test month 6
# 7.6M train chips, 1.5M test chips
# Stratified sampling: ensure all 6 defect types in both sets
```

---

## Step 10: Model Comparison

```python
# Model 1: Logistic Regression (OneVsRest)
# Macro F1: 0.61 (struggles with rare classes)

# Model 2: Random Forest (500 trees)
# Macro F1: 0.78

# Model 3: XGBoost (default, multiclass softmax)
# Macro F1: 0.83

# Model 4: LightGBM (default, multiclass)
# Macro F1: 0.85

# Model 5: SVM (RBF, OneVsRest)
# Macro F1: 0.74 (too slow on 7.6M rows -- 8 hours training)

# Per-class recall (LightGBM):
# OK:                 0.99
# Contamination:      0.92
# Alignment Error:    0.88
# Temp Excursion:     0.85
# Chemical Imbalance: 0.79
# Vibration Fault:    0.76
# Electrical Fault:   0.61  <-- only 25 cases/day, very hard
```

---

## Step 11: Handling Extreme Multiclass Imbalance

```python
# Problem: Electrical Fault has only 0.05% prevalence (25/day)
# With 7.6M training chips, that's ~3,800 electrical faults total
# But relative to 7.4M OK chips, the model barely learns it

# Strategy 1: Class weights
lgbm = LGBMClassifier(class_weight={
    'OK': 1, 'contamination': 50, 'alignment': 80, 'temp_excursion': 130,
    'chemical': 200, 'vibration': 270, 'electrical': 800
})
# Macro F1: 0.82 (electrical recall: 0.71, overall OK precision drops)

# Strategy 2: Hierarchical classification
# Level 1: OK vs Defective (binary) -- much more balanced (97.9% vs 2.1%)
# Level 2: Which defect? (6-class, only among defective chips)
# This separates the detection problem from the diagnosis problem

# Level 1: LightGBM binary classifier
# Recall: 0.94 (catches 94% of ALL defects combined)
# Precision: 0.71 (29% false alarms -- acceptable, they get manually inspected for $8)

# Level 2: LightGBM 6-class classifier (trained ONLY on defective chips)
# Now instead of 2.1% vs 97.9%, the classes are:
# Contamination: 38%, Alignment: 24%, Temp: 14%, Chemical: 10%, Vibration: 7%, Electrical: 2.4%
# Much more balanced! Electrical is still rare but 2.4% vs 0.05% is a massive improvement.
# Macro F1: 0.89 (electrical recall: 0.74)

# Combined hierarchical results:
# Overall detection recall: 0.94 (catches 94% of defects)
# Contamination diagnosis: 0.93
# Alignment diagnosis:     0.90
# Temp Excursion diagnosis: 0.87
# Chemical diagnosis:      0.83
# Vibration diagnosis:     0.80
# Electrical diagnosis:    0.70
```

**Expert insight:** The hierarchical approach works because it separates two fundamentally different tasks: "is this chip defective?" (binary, easier) vs "what type of defect?" (multiclass, but only among defective chips where features are more discriminative). The 6-class model trained on defective-only data sees 48x more electrical fault examples relative to its training set.

---

## Step 12-13: Tuning & Final Model

```python
# Tuned hierarchical system:
# Level 1 (defect detector): threshold lowered to 0.3 (catch more defects, accept more FPs)
#   Recall: 0.97, Precision: 0.58
# Level 2 (defect classifier): tuned per-class thresholds

# Cost-optimized decision:
# Flag as defective if P(defect) > 0.15 (very aggressive -- rather re-inspect than miss)
# Cost analysis:
#   Missed defects: 3% * 1050 defects/day = 31 missed * $120 = $3,720/day
#   False alarms: 820/day * $8 = $6,560/day
#   Total cost: $10,280/day
#
#   vs NO model: 1050 defects * $120 = $126,000/day
#   vs OPTICAL ONLY: current system costs $85,000/day (optical inspection + missed defects)
#   SAVINGS: $74,720/day = $27.3M/year
```

---

## Step 14-16: Evaluation, Explainability, Deployment

```python
# Per-chip explanation (for quality engineers):
# Chip #A4582: DEFECTIVE -- Contamination (confidence: 0.91)
# Evidence:
#   - particle_count_num_spikes = 7 (normal: 0-1)  ← main signal
#   - particle_count_max = 340 particles (normal: <50)
#   - Phase 2 humidity spike (+0.15 SHAP)
#   - Occurred during shift change (contamination cluster)

# Deployment:
# - Model runs on edge GPU at production line
# - Inference: 12ms per chip (well under 50ms limit)
# - Defective chips diverted to re-inspection station
# - Dashboard shows real-time defect rate per sensor, per shift
# - If defect rate exceeds 5% for any type, halt line + alert engineer
# - Retrain monthly with new labeled data from optical inspection
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Raw data | Feed 8,640 raw readings | Extracted 500 physics-motivated statistical features |
| Time series | Flatten and ignore order | Phase features, slopes, spike counts, FFT for vibration |
| Cross-sensor | Treat sensors independently | Chemical ratios, pressure-temp correlation, sync scores |
| Class imbalance | SMOTE everything | Hierarchical: detect first, then diagnose among defects |
| Rare classes | Accept low recall | Tuned per-class thresholds based on cost matrix |
| Feature engineering | Generic statistics | Domain-specific: FFT for vibration, ratios for chemistry |
| Evaluation | Overall accuracy | Per-defect-type recall + cost analysis |
| Deployment | Batch predictions | Real-time edge inference at 12ms per chip |

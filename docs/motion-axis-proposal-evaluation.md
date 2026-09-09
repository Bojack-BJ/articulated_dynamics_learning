# Motion-Axis Proposal Evaluation / 运动轴候选验证

## Contract / 协议

- Dataset order and sample cap: the same first 32 validation joints used by the aligned child-only pilot.
- Coordinates: checkpoint canonical center and scale.
- Quality: `observation_quality * track_quality`.
- Protocols: GT child assignment and predicted child slot assignment evaluated separately.
- Missing axes count as `90 deg` in the penalized mean.
- Selected proposal settings: group minimum 3 common tracks; track segments minimum 5 frames; visibility gaps up to 2 frames.

## Results / 结果

| Assignment | Method | Coverage | Axis mean | Axis median | Penalized mean | Revolute line |
|---|---:|---:|---:|---:|---:|---:|
| GT child | Group Kabsch | 81.2% | 28.00 deg | 10.50 deg | 39.62 deg | 0.0152 |
| GT child | Track voting | 87.5% | 24.04 deg | 15.48 deg | 32.29 deg | 0.0698 |
| GT child | Type-routed hybrid | 90.6% | 18.83 deg | 10.28 deg | 25.50 deg | 0.0152 |
| Predicted child slot | Group Kabsch | 75.0% | 15.17 deg | 8.73 deg | 33.88 deg | 0.0434 |
| Predicted child slot | Track voting | 93.8% | 28.88 deg | 19.11 deg | 32.70 deg | 0.1023 |
| Predicted child slot | Type-routed hybrid | 87.5% | 17.28 deg | 5.69 deg | 26.37 deg | 0.0434 |

## Interpretation / 结论

The methods are complementary rather than individually sufficient. Track voting is the stronger prismatic estimator (GT-child coverage `14/14`, mean `14.47 deg`), while group Kabsch is the appropriate revolute proposal family and gives better line localization. Low-support exact Kabsch fits and short circle arcs remain unstable. The routed hybrid improves both raw proposal methods but does not beat child-only analytic (`30/32`, `10.70 deg`, line `0.0185`).

两个方案互补但都不足以单独作为最终方法。Track voting 更适合 prismatic（GT-child `14/14` 有效，均值 `14.47 deg`）；group Kabsch 更适合作为 revolute 候选并具有更好的轴线定位。低支持度 Kabsch 精确拟合与短圆弧仍不稳定。按类型路由优于两个原始方案，但尚未超过 child-only analytic。

# Infrastructure inspection scenarios

The demo can illustrate camera monitoring around utility diversions, excavation works, material
storage and plant exclusion zones. Hong Kong programmes such as the Northern Metropolis provide
one useful context for discussing these workflows. The project remains a general construction
monitoring tool; its example policies have not been surveyed or calibrated at a real project site.

`examples/infrastructure-zones.yaml` contains two illustrative normalized polygons: an excavation
edge and a plant exclusion area. Adapt their position, required PPE and dwell policy to a fixed
camera view. Recheck the policy whenever the camera position, lens or worksite layout changes.

A useful demonstration follows a complete workflow:

1. Inspect whether the selected checkpoint supports the required PPE classes.
2. Define the restricted area and choose a temporal confirmation preset.
3. Replay licensed sample footage or connect an authorized camera.
4. Observe provisional findings, confirmed events and reconnect segments.
5. Review redacted evidence, record an operator decision and inspect delivery status.

The current rules cover supported PPE evidence and zone intrusion. They do not classify general
hazardous behaviour, falls, fatigue, vehicle interactions or excavation stability. A camera view
also cannot establish whether equipment is electrically isolated or whether a permit is valid.

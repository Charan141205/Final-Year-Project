CAPPIMU EULER ANGLE VALIDATION
==============================

Read-only scan of ex/ey/ez angle ranges, missing values, wrapped step sizes, and timestamps.
The +/-180, +/-90, +/-180 degree checks assume ex=X/roll, ey=Y/pitch, ez=Z/yaw provisionally.
The manufacturer datasheet specifies BWT901CL angle output in degrees and Z-Y-X order,
but the exact CAPPIMU column-to-angle export mapping must be confirmed before rotation.
No input recording or trained model is modified.

# UWB auto-calibration

This python package provides a multi-stage procedure to compute the positions of a set of UWB modules (anchors) placed in the 3D space based on inter-anchor range measurements. 

* The inter-anchor range measurements are reshaped in a `(n_anchors, n_samples, n_anchors)` array. For example, the position `(0, 5, 3)` contains the sixth measured range between `anchor_0` and `anchor_3`.
* A initial guess (i.e. 3D anchor positions) should be provided.
* The multi-stage procedure is based on two main stages:
    * Iterative least squares minimization.
    * Cost function minimization.

# Installation

Clone repository

```sh
git clone https://github.com/esauortiz/uwb_autocalibration/
```

Install requirements

```sh
pip install -r requirements.txt 
```

# Usage

Once the inter-anchor measurements have been taken (examples can be found [here](https://github.com/esauortiz/dwm1001_drivers/tree/main/autocalibration_datasets/uart)) and a nodes configuration file (see [nodes_cfg](https://github.com/esauortiz/dwm1001_drivers/tree/main/params/nodes_cfg)) is set, the auto-calibration is performed running:

```sh
python scripts/main.py <nodes_cfg_label> <n_samples> <n_discarded_samples>
```

Where:

* `nodes_cfg_label` is the label of the nodes configuration file.
* `n_samples` is the number of samples taken between one anchor and the others.
* `n_discarded_samples` number of samples placed at the beginning which will be discarded

The result of an auto-calibration is presented in the following figure:

![Result of an auto-calibration](https://github.com/esauortiz/uwb_autocalibration/blob/main/fig/autocalibration_campus_sport_new.png)

where all axis units are meters. The following table provides the error computed as the euclidean distance between the ground truth and the result of the multi-stage procedure. Errors on every axis is also provided:

| anchor\_id | error \[m\] | x\_error \[m\] | y\_error \[m\] | z\_error \[m\] |
| ---------- | ----------- | -------------- | -------------- | -------------- |
| DW009A     | 0.34        | 0.32           | 0.12           | \-0.00         |
| DW2D9C     | 0.39        | 0.32           | \-0.23         | \-0.00         |
| DW4848     | 0.39        | 0.33           | \-0.20         | \-0.00         |
| DW47FC     | 0.42        | 0.42           | \-0.04         | \-0.00         |
| DW0038     | 0.31        | 0.29           | 0.11           | 0.00           |
| DW4984     | 0.00        | 0.00           | 0.00           | 0.00           |
| DW4806     | 0.26        | 0.20           | 0.16           | \-0.00         |
| DW4814     | 0.36        | 0.25           | \-0.25         | 0.00           |
| DW43EB     | 0.00        | 0.00           | 0.00           | 0.00           |
| DW1632     | 0.09        | 0.01           | \-0.09         | 0.00           |
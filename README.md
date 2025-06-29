# Player Re-Identification in Sports Footage

## 📌 Overview
This project detects and tracks football players in a match video using YOLOv11 for detection and DeepSORT for tracking.
More Details about the project has been added in **report.pdf** uploaded in this repository.

## ⚙️ Setup Instructions

**Environment used:**  
- Python **3.10**
- Managed using **Anaconda**

To ensure reproducibility, please run this in an **Anaconda environment**.
Please note that the 'best.pt' yolov11 checkpoint file was too large to upload to github, hence its been uploaded to Google drive and the link is provided under the steps below.

1️- **Create and activate the environment** 

```bash
conda create -n player_reid python=3.10
conda activate player_reid
```
2- **install dependencies**

```bash
pip install -r requirements.txt
```
3- download yolov11 `best.pt` checkpoint from [this link](https://drive.google.com/file/d/11G9JIarMqkxUfF7-bF_ig9bFwbcxfLam/view?usp=sharing) and place it in the project root.

4- **run detect_and_track.py**

```bash
python detect_and_track.py
```

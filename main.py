from datahandler import EEGData, LabelPolicy, PreprocCfg, WindowCfg, EpochCfg

label_policy = LabelPolicy(
    aliases={"L_hand":"left","R_hand":"right","push":"left","pull":"right","rest":"null","relax":"null"},
    keep_labels=["left","right"],       # the labels we’ll train on
    others_action="null",               # "drop" or "null"
    null_name="null",
    fixed_ids={"left":1,"right":2,"null":3}  # stable IDs for reproducibility
)

pre = PreprocCfg(bandpass=(8,30), notch=50, reref="common", zscore=True)

win = WindowCfg(win_s=1.0, step_s=0.5, min_label_s=0.4, assign="majority")
epo = EpochCfg(tmin=0.0, tmax=2.0, baseline=None)  # or (-0.2,0.0)

data = EEGData.from_csv(
    data_files="data/raw/session01.csv",           # or a folder or list[str]
    events_files="data/raw/session01_events.csv",  # or None if embedded
    fs=250,
    channel_order=["Fp1","Fp2","C3","C4","O1","O2"],  # desired order/subset
    label_policy=label_policy,
    preproc=pre,
    window_cfg=win,
    epoch_cfg=epo,
)

# Continuous (C, T)
X, info = data.continuous()

# Windowed (N, C, W) and y (N,)
Xw, yw = data.windows()

# Epoched (N, C, W) and y (N,)
Xe, ye = data.epochs()

# Change anything → dependent views invalidate and recompute lazily:
data.set_window_cfg(WindowCfg(1.5, 0.75, 0.6, "center"))
Xw2, yw2 = data.windows()

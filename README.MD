

# LSTM
- src/ : CPU用
- srcgpu/ : GPU用(numpyをcupyに置換したもの)
- lib.py : Modelの定義、データ読み込み等
- optune.py : optunaでハイパーパラメータ探索し、trial_params.dumpへ出力
- optune_cross.py : データが少ないときに交差検定でハイパーパラメータ探索を行う
- readparams.py : trial_params.dumpに保存されたハイパーパラメータを用いて学習を行い、出来た分類器をmodel.npzへ出力する
- readmodel.py : trial_params.dumpとmodel.npzを用いて予測だけを行う

##ライブラリ
Chainer, Optuna, Cupy, Numpyあたり
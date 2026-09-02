set -e
cd /c/develop/dmu_adv_ai
# 백업 (Dual seed42 예측 보호)
cp ensemble_wf_predictions_seed42.csv _bak_seed42.csv
cp ensemble_wf_predictions.csv _bak_plain.csv
cp ensemble_wf_val_predictions.csv _bak_val.csv 2>/dev/null || true
# 60일 단독 앙상블 실행 (유의성 계산 포함)
LSTM_ARCH=single60 PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m app.models.ensemble_wf > scratch_ens60.txt 2>&1
ECODE=$?
# 복원 (반드시)
cp _bak_seed42.csv ensemble_wf_predictions_seed42.csv
cp _bak_plain.csv ensemble_wf_predictions.csv
cp _bak_val.csv ensemble_wf_val_predictions.csv 2>/dev/null || true
rm -f _bak_seed42.csv _bak_plain.csv _bak_val.csv
echo "RESTORED (exit=$ECODE)" >> scratch_ens60.txt

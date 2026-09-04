# 공개 체크포인트

학습 로그 전체(`logs/`)는 `.gitignore` 대상이다. 체크포인트가 100 iteration마다
5.2 MB씩 쌓여 한 런에 150개, 780 MB가 넘기 때문이다. 대신 **의미 있는 두 지점만**
여기에 복사해 둔다.

| 파일 | 원본 | 성격 |
|---|---|---|
| `qub_flat_model_15000.pt` | `Sep04_00-19-14_/model_15000.pt` | 학습 종료 시점(최종) |
| `qub_flat_model_6000.pt` | `Sep04_00-19-14_/model_6000.pt` | 지표가 가장 균형 잡힌 지점 |

## 어느 쪽을 쓸 것인가

15000이 최종이지만 **모든 지표에서 최고는 아니다**. 실측 비교:

| 지표 | model_6000 | model_15000 |
|---|---|---|
| 발바닥 기울기 L / R | **0.08° / 0.08°** | 0.36° / 0.02° |
| 6관절 미러 잔차 합 | 0.00015 rad² | **0.00012** |
| 좌우 발 앞뒤 차이 | 0.24 cm | **0.13 cm** |
| 좌우 보폭 차이 | **1.2%** | 8.8% |
| 뒤꿈치 최소 여유 | 4.23 cm | **5.02 cm** |
| 속도 추종 오차 | 7.1% | **6.5%** |
| 생존 (cmd 0.4) | 64/64 | 64/64 |

보폭 대칭과 발바닥 접지는 6000이, 관절 대칭과 발 위치는 15000이 낫다.
전 구간 추이는 [`../POLICY_PROGRESS.md`](../POLICY_PROGRESS.md) 참고.

시각적으로는 둘 다 구분이 어렵다 — 0.36°도 육안으로는 평평하다.

## 불러오기

`logs/qub_flat/qub_flat/<런이름>/model_<n>.pt` 경로를 기대하므로, 그 구조로 두고 실행한다.

```bash
mkdir -p logs/qub_flat/qub_flat/pretrained
cp checkpoints/qub_flat_model_15000.pt logs/qub_flat/qub_flat/pretrained/model_15000.pt

export PATH=$HOME/miniconda3/envs/kudos/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/kudos/lib:$LD_LIBRARY_PATH
export ROBOT_TYPE=QUB

python legged_gym/scripts/play.py --task=qub_flat \
    --load_run=pretrained --checkpoint=15000
```

측정은 같은 인자로 `legged_gym/scripts/diag/policy_report.py` 를 쓴다.

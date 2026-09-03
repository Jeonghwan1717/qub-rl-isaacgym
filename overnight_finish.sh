#!/bin/bash
# 학습 종료 -> 1000단위 전 구간 측정 -> 최종 상세 측정 -> 커밋/푸시 -> 전원 종료.
# Claude 세션과 무관하게 돌도록 setsid + nohup으로 띄운다.
#
# 설계상 주의:
#  - 순서가 중요하다. 푸시가 끝나기 전에 꺼지면 결과가 GitHub에 안 올라간다.
#  - policy_report.py(통합, 미검증)가 실패해도 결과가 남도록, 최종 체크포인트는
#    이미 검증된 개별 스크립트로 한 번 더 측정한다.
#  - 각 측정에 timeout을 걸어 하나가 멈춰도 전체가 밤새 걸리지 않게 한다.

set -u
REPO=/home/kim/kudos_ws/rl_library/tron1-rl-isaacgym
RUN=Sep04_00-19-14_
LOG=/home/kim/overnight_finish.log   # /tmp은 재부팅에 지워지므로 홈에 남긴다

export PATH=/home/kim/miniconda3/envs/kudos/bin:$PATH
export LD_LIBRARY_PATH=/home/kim/miniconda3/envs/kudos/lib:$LD_LIBRARY_PATH
export ROBOT_TYPE=QUB
cd "$REPO" || exit 1

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== 학습 종료 대기 시작 ==="
while pgrep -f "legged_gym/scripts/train.py" > /dev/null; do
    sleep 60
done
log "학습 종료 감지"

CKDIR="logs/qub_flat/qub_flat/$RUN"
LAST=$(ls "$CKDIR"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' \
       | sort -n | tail -1)

if [ -z "$LAST" ]; then
    log "!! 체크포인트를 찾지 못함 - 측정 생략"
else
log "최종 체크포인트 model_$LAST"

# ---------- 1) 1000단위 추이 표 ----------
PROG=POLICY_PROGRESS.md
{
    echo "# 학습 경과별 보행 정책 지표"
    echo
    echo "런 \`$RUN\` 의 1000 iteration 단위 체크포인트를 같은 조건으로 측정한 결과."
    echo
    echo "측정 조건: cmd_vx 0.4 · 64 env · 도메인 랜덤화/관측 노이즈 off ·"
    echo "명령 램프업 후 10초. 전 구간 생존한 개체만 통계에 포함."
    echo
    echo "재현: \`python legged_gym/scripts/diag/policy_report.py --task=qub_flat \\\\\`"
    echo "\`  --load_run=$RUN --checkpoint=<n> --headless\`"
    echo
    echo "| iter | 생존 | 실제 vx | 오차% | 발바닥 L° | 발바닥 R° | 6관절합 rad² | 최대관절° | 앞뒤차 cm | 보폭차% | 뒤꿈치여유 cm | torso_yaw° |"
    echo "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
} > "$PROG"

for n in $(seq 1000 1000 15000); do
    [ -f "$CKDIR/model_$n.pt" ] || continue
    ROW=$(timeout 900 python -u legged_gym/scripts/diag/policy_report.py \
          --task=qub_flat --load_run="$RUN" --checkpoint="$n" --headless --tsv \
          2>/dev/null | grep -E "^$n	" | tail -1)
    if [ -n "$ROW" ]; then
        echo "$ROW" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
             $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' >> "$PROG"
        log "model_$n 측정 완료"
    else
        echo "| $n | 측정 실패 | | | | | | | | | | |" >> "$PROG"
        log "!! model_$n 측정 실패"
    fi
done

{
    echo
    echo "## 열 설명"
    echo
    echo "- **발바닥 L/R**: 접촉 중 발바닥과 땅 사이 각도. 0에 가까울수록 발바닥 전체가 닿는다."
    echo "  골반이 아니라 월드 기준으로 재며, pitch와 roll을 함께 본다."
    echo "- **6관절합**: 다리 6관절쌍 미러 잔차의 제곱합. \`joint_symmetry\` 리워드의 raw 값."
    echo "- **최대관절**: 6쌍 중 가장 크게 어긋난 관절의 잔차."
    echo "- **앞뒤차**: 좌우 발의 골반 기준 앞뒤 평균 위치 차이."
    echo "- **뒤꿈치여유**: 두 발 충돌 박스 사이 최소 거리. 발끝이 벌어지면 뒤꿈치가 먼저 만난다."
    echo "- **torso_yaw**: 상체 비틀림의 DC 성분. 좌우 흔들림(AC)이 아니라 한쪽으로 고정된 양."
} >> "$PROG"
log "추이 표 생성 완료"

# ---------- 2) 최종 체크포인트 상세 (검증된 개별 스크립트) ----------
OUT=RESULTS.md
{
    echo "# QUB 보행 학습 최종 결과"
    echo
    echo "최종 정책: \`$RUN/model_$LAST\`"
    echo
    echo "측정 조건은 \`POLICY_PROGRESS.md\` 와 동일. 경과별 추이는 그쪽 참고."
    echo
    echo "생성 시각: $(date '+%F %T')"
    echo
} > "$OUT"

for s in foot_flatness heel_clearance joint_sym_check step_asym; do
    echo "## $s" >> "$OUT"
    echo '```' >> "$OUT"
    timeout 900 python -u "legged_gym/scripts/diag/$s.py" --task=qub_flat \
        --load_run="$RUN" --checkpoint="$LAST" --headless 2>/dev/null \
        | sed -n '/^=\{10,\}/,$p' >> "$OUT"
    echo '```' >> "$OUT"
    echo >> "$OUT"
    log "최종 $s 완료"
done
fi

# ---------- 3) 커밋 / 푸시 ----------
# 소스 .py 전부와 보고서를 담는다. logs/는 .gitignore라 체크포인트는 안 들어간다.
git add -A -- '*.py' RESULTS.md POLICY_PROGRESS.md \
    legged_gym/scripts/diag overnight_finish.sh >> "$LOG" 2>&1

if git diff --cached --quiet; then
    log "커밋할 변경 없음 - 푸시 생략"
else
    git commit -q -m "학습 결과 기록: $RUN (최종 model_${LAST:-?})

15000 iteration 학습을 마친 뒤 1000 단위 체크포인트를 모두 같은 조건으로 측정했다.
POLICY_PROGRESS.md 에 경과별 추이 표를, RESULTS.md 에 최종 정책의 상세 측정을 담았다.
측정 조건과 각 지표의 의미는 두 파일 상단과 diag/README.md 참고.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" >> "$LOG" 2>&1
    log "커밋 완료"
    if git push myfork master >> "$LOG" 2>&1; then
        log "푸시 성공"
    else
        log "!! 푸시 실패 - 커밋은 로컬에 남아있으니 아침에 수동 푸시할 것"
    fi
fi

log "sync 후 전원 종료"
sync
sleep 10
systemctl poweroff

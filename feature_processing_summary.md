# CatBoost 피처 전처리 점검표

현재 `train_cat_single.py`에 적용되어 있는 최신 피처 전처리 방식을 4가지 그룹으로 나누어 정리한 표입니다.

## 1. 수식 기반 파생 변수 (Derived Features)

| 피처명 | 설명 | 전처리 방식 및 식 |
|---|---|---|
| `same_hand` | 투타 매치업 (동일 손 여부) | `pitcher_hand == batter_hand` (True=1, False=0) |
| `count_advantage` | 볼카운트 투수 유리도 | `(strikes_before - balls_before + 3) / 5.0` |
| `win_expectancy_pitcher_team` | 투수 팀 기준 기대 승률 압박감 | `top_bottom`이 'T'(초)이면 `home_win_expectancy`, 아니면 `away_win_expectancy`를 가져온 뒤, 50을 기준으로 절대값 거리를 구함: `(x - 50).abs() / 50.0` |
| `re24` | 아웃 및 주자 상황별 기대 득점 | (1루, 2루, 3루, 아웃카운트) 상태 코드를 고정 RE24 테이블(24개 상태)에 매핑 (연속형 변수로 취급) |
| `pitcher_momentum_1_vs_5` | 제구 성공률 단기 모멘텀 | `직전 1경기 성공률` - `직전 5경기 성공률.fillna(0)` |
| `pitcher_condition_vs_baseline` | 제구 성공률 컨디션 | `직전 3경기 성공률` - `누적 성공률` |
* **제거됨**: `pitcher_middle_momentum_1_vs_5`, `pitcher_middle_condition_vs_baseline` (실투율 모멘텀 관련 피처 제거됨)

## 2. 수치형 스케일링 변수 (Scaled Numeric Features)

> [!TIP]
> 앞서 논의했듯, 트리 모델(CatBoost)에서는 이러한 선형 스케일링(나누기, 로그 등)이 성능 향상에 큰 기여를 하지는 않지만, 기존 딥러닝 코드와의 호환성을 위해 유지되어 있는 부분들입니다.

| 피처명 | 원본 설명 | 전처리 방식 및 식 |
|---|---|---|
| `balls_before` | 볼 카운트 | `/ 3.0` |
| `strikes_before` | 스트라이크 카운트 | `/ 2.0` |
| `inning` | 이닝 | `(inning - 1).clip(upper=8) / 8.0` |
| `li` | 상황 중요도 | `.clip(upper=2) / 2.0` |
| `asof_pitcher_n` | 투수 누적 투구수 | `np.log1p()` (로그 변환) |
| `asof_pitcher_pitchmix_n` | 투수 구종 표본 수 | `np.log1p()` (로그 변환) |
* **제거됨**: 
  - `outs_before` (단일 피처로는 삭제되고 `base_out_state`와 `re24`로 통합됨)
  - `score_diff_pitcher_team`, `asof_batter_n` (피처 중요도 및 모델 간소화를 위해 사용 피처에서 제외)

## 3. 원본 유지 수치형 변수 (Raw Numeric Features)

| 카테고리 | 포함 피처명 | 전처리 방식 |
|---|---|---|
| **시간 변수** | `season`, `game_month` | 없음 (이전의 sin/cos 주기 변환 삭제됨) |
| **비율(Rate) 변수** | `asof_pitcher_success_rate`, `asof_pitcher_strike_rate`, `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate`, `asof_pitcher_reverse_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_fastball_rate` | 없음 |
| **과거 이력(Rate) 변수** | `asof_batter_success_rate`, `asof_pitcher_prev1_game_success_rate`, `asof_pitcher_prev3_game_success_rate`, `asof_pitcher_prev5_game_success_rate` | 없음 (결측치는 CatBoost가 분기 시 자체적으로 최적의 방향으로 할당함) |
* **제거됨**: 주자 개별 변수(`runner_on_1b`, `runner_on_2b`, `runner_on_3b`), `asof_pitcher_middle_rate` 및 모든 `middle_rate` 계열 과거 이력 변수

## 4. 범주형 변수 (Categorical Features)

> [!NOTE]
> 아래 변수들은 모두 `cat_cols` 리스트에 포함되어 훈련 시 `cat_features` 인자로 넘어갑니다. 모델 내부에서 Target Encoding과 조합(Combination)을 통해 강력한 교호작용을 파악하는 핵심 피처들입니다.

| 피처명 | 설명 | 전처리 방식 |
|---|---|---|
| `pitcher_team_id` | 투수 소속 팀 ID | `.fillna('MISSING').astype(str)` |
| `batter_team_id` | 타자 소속 팀 ID | `.fillna('MISSING').astype(str)` |
| `pitcher_hand` | 투수 투구 손 | `.fillna('MISSING').astype(str)` |
| `batter_hand` | 타자 타격 손 | `.fillna('MISSING').astype(str)` |
| `game_type` | 경기 유형 | `.fillna('MISSING').astype(str)` |
| `base_out_state` | 24가지 주자/아웃 상황 | 주자(1,2,3루)와 아웃카운트 조합을 하나의 문자열로 결합하여 24가지의 고유 상황 범주 생성 |
* **제거됨**: `pitcher_id`, `batter_id` (과적합 방지 목적), `top_bottom` (피처 중요도 0으로 제외)
